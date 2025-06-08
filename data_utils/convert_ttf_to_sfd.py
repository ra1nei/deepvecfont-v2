import fontforge  # noqa
import os
import multiprocessing as mp
import argparse

# Global lock for safe printing across processes
print_lock = mp.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def convert_mp(opts):
    charset_file_path = os.path.join(opts.charset_path, f"{opts.language}.txt")

    if not os.path.exists(charset_file_path):
        safe_print(f"Charset file for language '{opts.language}' not found at {charset_file_path}")
        return

    with open(charset_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    charset = ''.join([line.strip() for line in lines if line.strip()])
    charset_lenw = len(str(len(charset)))

    fonts_file_path = os.path.join(opts.ttf_path, opts.language)
    sfd_path = os.path.join(opts.sfd_path, opts.language)

    ttf_fnames = []
    for root, dirs, files in os.walk(os.path.join(fonts_file_path, opts.split)):
        ttf_fnames.extend(files)

    if not ttf_fnames:
        safe_print(f"No font files found in {fonts_file_path}")
        return

    font_num = len(ttf_fnames)
    num_batches = opts.num_batches if opts.num_batches else max(mp.cpu_count() - 2, 1)
    font_num_per_batch = font_num // num_batches + 1

    manager = mp.Manager()
    error_fonts = manager.list()

    def process(batch_id, font_num_p_batch, error_fonts):
        for i in range(batch_id * font_num_p_batch, (batch_id + 1) * font_num_p_batch):
            if i >= font_num:
                break

            font_id = ttf_fnames[i].split('.')[0]
            split = opts.split
            font_name = ttf_fnames[i]
            font_file_path = os.path.join(fonts_file_path, split, font_name)

            try:
                cur_font = fontforge.open(font_file_path)
                cur_font.encoding = "UnicodeFull"
            except Exception as e:
                safe_print(f"[Batch {batch_id}] ❌ Không thể mở font {font_name}: {e}")
                error_fonts.append(font_name)
                continue

            target_dir = os.path.join(sfd_path, split, f"{font_id}")
            os.makedirs(target_dir, exist_ok=True)

            for char_id, char in enumerate(charset):
                safe_print('=======================================================')
                safe_print(f"[Batch {batch_id}] Char: {char} | Unicode: {ord(char)} | Font: {font_name}")

                try:
                    char_file = os.path.join(target_dir, f'{font_id}_{char_id:0{charset_lenw}}.txt')
                    with open(char_file, 'w') as char_description:
                        cur_font.encoding = 'UnicodeFull'
                        cur_font.selection.select((ord(char)))
                        cur_font.copy()

                        new_font_for_char = fontforge.font()
                        new_font_for_char.encoding = 'UnicodeFull'
                        new_font_for_char.selection.select('A')
                        new_font_for_char.paste()

                        new_font_for_char.fontname = f"{font_id}_{font_name}"
                        sfd_file = os.path.join(target_dir, f'{font_id}_{char_id:0{charset_lenw}}.sfd')
                        new_font_for_char.save(sfd_file)

                        char_description.write(f"{char}\n")
                        char_description.write(f"{new_font_for_char[char].width}\n")
                        char_description.write(f"{new_font_for_char[char].vwidth}\n")
                        char_description.write(f"{char_id:0{charset_lenw}}\n")
                        char_description.write(f"{font_id}")
                except Exception as e:
                    safe_print(f"[Batch {batch_id}] ⚠️ Lỗi khi xử lý glyph '{char}' trong font '{font_name}': {e}")
                    error_fonts.append(font_name)
                    continue
                safe_print('=======================================================\n')

            cur_font.close()

    processes = [mp.Process(target=process, args=(pid, font_num_per_batch, error_fonts)) for pid in range(num_batches)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    error_file_path = '/kaggle/working/error_fonts.txt'
    with open(error_file_path, 'w') as f:
        unique_fonts = set(error_fonts)
        f.write(f"Total number of error fonts: {len(unique_fonts)}\n")
        for font in unique_fonts:
            f.write(f"{font}\n")

    safe_print(f"✅ Error fonts saved to {error_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert ttf fonts to sfd fonts")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn', 'vie'], help="Language for charset")
    parser.add_argument("--charset_path", type=str, default='/kaggle/input/font-eng-mini/deepvecfont-v2-minidataset-eng/char_set', help="Path to charset files")
    parser.add_argument("--ttf_path", type=str, default='../data/font_ttfs', help="Path to TTF font files")
    parser.add_argument('--sfd_path', type=str, default='../data/font_sfds', help="Path to save SFD files")
    parser.add_argument('--split', type=str, default='train', help="Data split (train/test)")
    parser.add_argument('--num_batches', type=int, default=None, help="Number of batches to split fonts into")

    opts = parser.parse_args()
    convert_mp(opts)

if __name__ == "__main__":
    main()
