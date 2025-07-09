import os
import random
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from dataloader import get_loader
from models import util_funcs
from models.model_main import ModelMain
from options import get_parser_main_model
from data_utils.svg_utils import render

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_main_model(opts):
    setup_seed(1111)
    dir_exp = os.path.join("./experiments", opts.name_exp)
    dir_sample = os.path.join(dir_exp, "samples")
    dir_ckpt = os.path.join(dir_exp, "checkpoints")
    dir_log = os.path.join(dir_exp, "logs")
    logfile_train = open(os.path.join(dir_log, "train_loss_log.txt"), 'w')
    logfile_val = open(os.path.join(dir_log, "val_loss_log.txt"), 'w')

    train_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size, opts.mode, num_samples=opts.num_train_samples)
    val_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size_val, 'test', num_samples=opts.num_test_samples)

    model_main = ModelMain(opts)

    if torch.cuda.is_available() and opts.multi_gpu:
        model_main = torch.nn.DataParallel(model_main)
    
    model_main.cuda()
    
    parameters_all = [{"params": model_main.img_encoder.parameters()}, {"params": model_main.img_decoder.parameters()},
                        {"params": model_main.modality_fusion.parameters()}, {"params": model_main.transformer_main.parameters()},
                        {"params": model_main.transformer_seqdec.parameters()}]
    
    optimizer = Adam(parameters_all, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    
    # --- CHECKPOINT BLOCK ---
    start_batch = 0 # Default start batch
    if opts.resume_ckpt is not None:
        print(f"Loading checkpoint from {opts.resume_ckpt} to resume training...")
        try:
            checkpoint = torch.load(opts.resume_ckpt)
            if opts.multi_gpu:
                model_main.module.load_state_dict(checkpoint['model'])
            else:
                model_main.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['opt'])

            opts.init_epoch = checkpoint['n_epoch'] + 1 

            start_epoch = checkpoint['n_epoch'] + 1 # Start from the next epoch
            start_batch = checkpoint['n_iter'] # Can be used if you want to resume mid-epoch, but often easier to just start a new epoch
            # If your scheduler state is also saved, load it here.
            # E.g., if you saved 'scheduler': scheduler.state_dict()
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Resuming training from Epoch {opts.init_epoch}, Batch {checkpoint['n_iter']}")
            start_batch = checkpoint['n_iter']

        except FileNotFoundError:
            print(f"Checkpoint file not found: {opts.resume_ckpt}. Starting training from scratch.")
        except KeyError as e:
            print(f"Error loading checkpoint: Missing key {e}. Check if the checkpoint structure matches. Starting training from scratch.")
        except Exception as e:
            print(f"An unexpected error occurred while loading checkpoint: {e}. Starting training from scratch.")

    if opts.tboard:
        writer = SummaryWriter(dir_log)

    for epoch in range(opts.init_epoch, opts.n_epochs):
        # Accumulators for epoch-wise logging
        total_train_loss = 0.0
        total_img_l1_loss = 0.0
        total_img_pt_c_loss = 0.0
        total_svg_total_loss = 0.0
        total_svg_cmd_loss = 0.0
        total_svg_args_loss = 0.0
        total_svg_smooth_loss = 0.0
        total_svg_aux_loss = 0.0
        total_kl_loss = 0.0
        
        # Variables to store the last sample for saving image
        last_ret_dict = None

        for idx, data in enumerate(train_loader):
            for key in data: data[key] = data[key].cuda()
            ret_dict, loss_dict = model_main(data)

            loss = opts.loss_w_l1 * loss_dict['img']['l1'] + opts.loss_w_pt_c * loss_dict['img']['vggpt'] + opts.kl_beta * loss_dict['kl'] \
                     + loss_dict['svg']['total'] + loss_dict['svg_para']['total']

            # perform optimization
            optimizer.zero_grad()
            loss.backward()      
            optimizer.step()
            
            # Accumulate losses for epoch-wise average
            total_train_loss += loss.item()
            total_img_l1_loss += opts.loss_w_l1 * loss_dict['img']['l1'].item()
            total_img_pt_c_loss += opts.loss_w_pt_c * loss_dict['img']['vggpt']
            total_svg_total_loss += loss_dict['svg']['total'].item() + loss_dict['svg_para']['total'].item() # Sum of both parallel and non-parallel SVG total loss
            total_svg_cmd_loss += opts.loss_w_cmd * (loss_dict['svg']['cmd'].item() + loss_dict['svg_para']['cmd'].item())
            total_svg_args_loss += opts.loss_w_args * (loss_dict['svg']['args'].item() + loss_dict['svg_para']['args'].item())
            total_svg_smooth_loss += opts.loss_w_smt * (loss_dict['svg']['smt'].item() + loss_dict['svg_para']['smt'].item())
            total_svg_aux_loss += opts.loss_w_aux * (loss_dict['svg']['aux'].item() + loss_dict['svg_para']['aux'].item())
            total_kl_loss += opts.kl_beta * loss_dict['kl'].item()

            # Store the last ret_dict for image sampling at the end of the epoch
            if idx == len(train_loader) - 1: # Only store the last one
                last_ret_dict = ret_dict

        # --- Actions performed once per epoch ---
        batches_done_at_epoch_end = (epoch + 1) * len(train_loader) # This will be the 'n_iter' for checkpoint

        # Calculate average losses for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_img_l1_loss = total_img_l1_loss / len(train_loader)
        avg_img_pt_c_loss = total_img_pt_c_loss / len(train_loader)
        avg_svg_total_loss = total_svg_total_loss / len(train_loader)
        avg_svg_cmd_loss = total_svg_cmd_loss / len(train_loader)
        avg_svg_args_loss = total_svg_args_loss / len(train_loader)
        avg_svg_smooth_loss = total_svg_smooth_loss / len(train_loader)
        avg_svg_aux_loss = total_svg_aux_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)

        # Log training loss per epoch
        message = (
            f"Epoch: {epoch}/{opts.n_epochs}, Avg_Loss: {avg_train_loss:.6f}, "
            f"Avg_img_l1: {avg_img_l1_loss:.6f}, Avg_img_pt_c: {avg_img_pt_c_loss:.6f}, "
            f"Avg_svg_total: {avg_svg_total_loss:.6f}, Avg_svg_cmd: {avg_svg_cmd_loss:.6f}, "
            f"Avg_svg_args: {avg_svg_args_loss:.6f}, Avg_svg_smooth: {avg_svg_smooth_loss:.6f}, "
            f"Avg_svg_aux: {avg_svg_aux_loss:.6f}, Avg_kl: {avg_kl_loss:.6f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        logfile_train.write(message + '\n')
        print(message)
        
        if opts.tboard:
            writer.add_scalar('Loss/avg_loss_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/avg_img_l1_epoch', avg_img_l1_loss, epoch)
            writer.add_scalar('Loss/avg_img_pt_c_epoch', avg_img_pt_c_loss, epoch)
            writer.add_scalar('Loss/avg_svg_total_epoch', avg_svg_total_loss, epoch)
            writer.add_scalar('Loss/avg_svg_cmd_epoch', avg_svg_cmd_loss, epoch)
            writer.add_scalar('Loss/avg_svg_args_epoch', avg_svg_args_loss, epoch)
            writer.add_scalar('Loss/avg_svg_smooth_epoch', avg_svg_smooth_loss, epoch)
            writer.add_scalar('Loss/avg_svg_aux_epoch', avg_svg_aux_loss, epoch)
            writer.add_scalar('Loss/avg_kl_epoch', avg_kl_loss, epoch)
            # Use the last sample for image logging
            if last_ret_dict:
                writer.add_image('Images/trg_img_epoch', last_ret_dict['img']['trg'][0], epoch)
                writer.add_image('Images/img_output_epoch', last_ret_dict['img']['out'][0], epoch)


        # Sample and save images per epoch
        if opts.freq_sample > 0 and (epoch + 1) % opts.freq_sample == 0: # Check if it's time to sample based on epoch
            if last_ret_dict: # Ensure we have a sample from the last batch
                img_sample = torch.cat((last_ret_dict['img']['trg'].data, last_ret_dict['img']['out'].data), -2)
                save_file = os.path.join(dir_sample, f"train_epoch_{epoch}.png") # Name by epoch
                save_image(img_sample, save_file, nrow=8, normalize=True)
                print(f"Saved sample image for epoch {epoch} to {save_file}")
            else:
                print(f"Warning: No sample image to save for epoch {epoch}. last_ret_dict was None.")

        # Validate and log validation loss per epoch
        if opts.freq_val > 0 and (epoch + 1) % opts.freq_val == 0: # Check if it's time to validate based on epoch
            with torch.no_grad():
                model_main.eval() # Set model to evaluation mode
                val_loss = {'img':{'l1':0.0, 'vggpt':0.0}, 'svg':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0, 'smt':0.0}, # Added smt here
                            'svg_para':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0, 'smt':0.0}} # Added smt here
                
                # Accumulate validation losses
                for val_idx, val_data in enumerate(val_loader):
                    for key in val_data: val_data[key] = val_data[key].cuda()
                    ret_dict_val, loss_dict_val = model_main(val_data, mode='val')
                    
                    # Accumulate for SVG losses (both parallel and non-parallel)
                    val_loss['svg']['total'] += loss_dict_val['svg']['total'].item()
                    val_loss['svg']['cmd'] += loss_dict_val['svg']['cmd'].item()
                    val_loss['svg']['args'] += loss_dict_val['svg']['args'].item()
                    val_loss['svg']['aux'] += loss_dict_val['svg']['aux'].item()
                    val_loss['svg']['smt'] += loss_dict_val['svg']['smt'].item()

                    val_loss['svg_para']['total'] += loss_dict_val['svg_para']['total'].item()
                    val_loss['svg_para']['cmd'] += loss_dict_val['svg_para']['cmd'].item()
                    val_loss['svg_para']['args'] += loss_dict_val['svg_para']['args'].item()
                    val_loss['svg_para']['aux'] += loss_dict_val['svg_para']['aux'].item()
                    val_loss['svg_para']['smt'] += loss_dict_val['svg_para']['smt'].item()

                    # Accumulate for image losses
                    val_loss['img']['l1'] += loss_dict_val['img']['l1'].item()
                    val_loss['img']['vggpt'] += loss_dict_val['img']['vggpt'].item()

                # Calculate average validation losses
                num_val_batches = len(val_loader)
                if num_val_batches > 0: # Avoid division by zero if val_loader is empty
                    for loss_cat in ['img', 'svg', 'svg_para']:
                        for key in val_loss[loss_cat]: # Iterate over keys directly
                            val_loss[loss_cat][key] /= num_val_batches 

                if opts.tboard:
                    writer.add_scalar(f'VAL/loss_img_l1_epoch', val_loss['img']['l1'], epoch)
                    writer.add_scalar(f'VAL/loss_img_vggpt_epoch', val_loss['img']['vggpt'], epoch)
                    writer.add_scalar(f'VAL/loss_svg_total_epoch', val_loss['svg']['total'] + val_loss['svg_para']['total'], epoch) # Sum both for overall SVG loss
                    writer.add_scalar(f'VAL/loss_svg_cmd_epoch', val_loss['svg']['cmd'] + val_loss['svg_para']['cmd'], epoch)
                    writer.add_scalar(f'VAL/loss_svg_args_epoch', val_loss['svg']['args'] + val_loss['svg_para']['args'], epoch)
                    writer.add_scalar(f'VAL/loss_svg_aux_epoch', val_loss['svg']['aux'] + val_loss['svg_para']['aux'], epoch)
                    writer.add_scalar(f'VAL/loss_svg_smt_epoch', val_loss['svg']['smt'] + val_loss['svg_para']['smt'], epoch)
                    
                val_msg = (
                    f"Epoch: {epoch}/{opts.n_epochs}, "
                    f"Val loss img l1: {val_loss['img']['l1']: .6f}, "
                    f"Val loss img pt: {val_loss['img']['vggpt']: .6f}, "
                    f"Val loss total SVG: {(val_loss['svg']['total'] + val_loss['svg_para']['total']): .6f}, " # Sum both
                    f"Val loss cmd SVG: {(val_loss['svg']['cmd'] + val_loss['svg_para']['cmd']): .6f}, "
                    f"Val loss args SVG: {(val_loss['svg']['args'] + val_loss['svg_para']['args']): .6f}, "
                    f"Val loss smooth SVG: {(val_loss['svg']['smt'] + val_loss['svg_para']['smt']): .6f}, "
                    f"Val loss aux SVG: {(val_loss['svg']['aux'] + val_loss['svg_para']['aux']): .6f}"
                )

                logfile_val.write(val_msg + "\n")
                print(val_msg)
            model_main.train() # Set model back to train mode after validation

        scheduler.step() # Learning rate scheduler step (usually per epoch)

        # Save checkpoint per epoch (this logic remains mostly the same, but n_iter will be epoch-end batches_done)
        if (epoch + 1) % opts.freq_ckpt == 0: # Changed to (epoch + 1) to align with epoch number
            if opts.multi_gpu:
                torch.save({'model':model_main.module.state_dict(), 'opt':optimizer.state_dict(), 'n_epoch':epoch, 'n_iter':batches_done_at_epoch_end}, f'{dir_ckpt}/{epoch+1}.ckpt') # Name by epoch
            else:
                torch.save({'model':model_main.state_dict(), 'opt':optimizer.state_dict(), 'n_epoch':epoch, 'n_iter':batches_done_at_epoch_end}, f'{dir_ckpt}/{epoch+1}.ckpt') # Name by epoch
            print(f"Saved checkpoint for epoch {epoch+1} to {dir_ckpt}/{epoch+1}.ckpt")


    logfile_train.close()
    logfile_val.close()

def backup_code(name_exp):
    os.makedirs(os.path.join('experiments', name_exp, 'code'), exist_ok=True)
    shutil.copy('models/transformers.py', os.path.join('experiments', name_exp, 'code', 'transformers.py') )
    shutil.copy('models/model_main.py', os.path.join('experiments', name_exp, 'code', 'model_main.py'))
    shutil.copy('models/image_encoder.py', os.path.join('experiments', name_exp, 'code', 'image_encoder.py'))
    shutil.copy('models/image_decoder.py', os.path.join('experiments', name_exp, 'code', 'image_decoder.py'))
    shutil.copy('./train.py', os.path.join('experiments', name_exp, 'code', 'train.py'))
    shutil.copy('./options.py', os.path.join('experiments', name_exp, 'code', 'options.py'))

def train(opts):
    if opts.model_name == 'main_model':
        train_main_model(opts)
    elif opts.model_name == 'others':
        train_others(opts)
    else:
        raise NotImplementedError

def main():
    
    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + '_' + opts.model_name
    os.makedirs("./experiments", exist_ok=True)
    debug = True
    # Create directories
    experiment_dir = os.path.join("./experiments", opts.name_exp)
    backup_code(opts.name_exp)
    os.makedirs(experiment_dir, exist_ok=debug)  # False to prevent multiple train run by mistake
    os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    print(f"Training on experiment {opts.name_exp}...")
    # Dump options
    with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
        for key, value in vars(opts).items():
            f.write(str(key) + ": " + str(value) + "\n")
    train(opts)

if __name__ == "__main__":
    main()
