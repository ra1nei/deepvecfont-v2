import fontforge  # noqa
import os
import multiprocessing as mp
import argparse

# conda deactivate
# apt install python3-fontforge

def convert_mp(opts):
    """Using multiprocessing to convert all fonts to sfd files"""
    # Kiểm tra và mở tệp charset tương ứng với ngôn ngữ
    charset_file_path = os.path.join(opts.charset_path, f"{opts.language}.txt")
    
    # Kiểm tra xem tệp charset có tồn tại không
    if not os.path.exists(charset_file_path):
        print(f"Charset file for language '{opts.language}' not found at {charset_file_path}")
        return
    
    charset = open(charset_file_path, 'r').read()
    charset_lenw = len(str(len(charset)))
    fonts_file_path = os.path.join(opts.ttf_path, opts.language)  # opts.ttf_path, opts.language
    sfd_path = os.path.join(opts.sfd_path, opts.language)

    # Initialize ttf_fnames as an empty list
    ttf_fnames = []

    for root, dirs, files in os.walk(os.path.join(fonts_file_path, opts.split)):
        ttf_fnames.extend(files)  # Collect all files into the list

    if not ttf_fnames:
        print(f"No font files found in {fonts_file_path}")
        return  # Exit early if no fonts are found

    font_num = len(ttf_fnames)
    process_num = mp.cpu_count() - 2
    font_num_per_process = font_num // process_num + 1

    error_fonts = set()

    def process(process_id, font_num_p_process):
        for i in range(process_id * font_num_p_process, (process_id + 1) * font_num_p_process):
            if i >= font_num:
                break

            font_id = ttf_fnames[i].split('.')[0]
            split = opts.split
            font_name = ttf_fnames[i]

            font_file_path = os.path.join(fonts_file_path, split, font_name)
            try:
                cur_font = fontforge.open(font_file_path)
            except Exception as e:
                print(f"Không thể mở font {font_name}: {e}")
                error_fonts.update([font_name])  # Lưu lại tên font lỗi
                continue  # Tiếp tục với font khác nếu không thể mở

            target_dir = os.path.join(sfd_path, split, "{}".format(font_id))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for char_id, char in enumerate(charset):
                try:
                    char_description = open(os.path.join(target_dir, '{}_{num:0{width}}.txt'.format(font_id, num=char_id, width=charset_lenw)), 'w')

                    cur_font.encoding = 'UnicodeFull'  # ✅ Bổ sung dòng này
                    cur_font.selection.select(ord(char))  # ✅ Unicode-safe selection
                    cur_font.copy()

                    new_font_for_char = fontforge.font()
                    new_font_for_char.encoding = 'UnicodeFull'
                    new_font_for_char.selection.select(("unicode", ord(char)))  # ✅ Unicode-safe selection
                    new_font_for_char.paste()

                    new_font_for_char.fontname = "{}_{}".format(font_id, font_name)

                    new_font_for_char.save(os.path.join(target_dir, '{}_{num:0{width}}.sfd'.format(font_id, num=char_id, width=charset_lenw)))

                    char_description.write(str(ord(char)) + '\n')
                    char_description.write(str(new_font_for_char[char].width) + '\n')
                    char_description.write(str(new_font_for_char[char].vwidth) + '\n')
                    char_description.write('{num:0{width}}'.format(num=char_id, width=charset_lenw) + '\n')
                    char_description.write('{}'.format(font_id))
                    char_description.close()

                except Exception as e:
                    print(f"Lỗi khi xử lý glyph {char} trong font {font_name}: {e}")
                    error_fonts.update([font_name])  # Lưu lại tên font lỗi
                    continue  # Tiếp tục với glyph khác nếu có lỗi

            cur_font.close()

    processes = [mp.Process(target=process, args=(pid, font_num_per_process)) for pid in range(process_num)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Save error information to a text file
    error_file_path = '/kaggle/working/error_fonts.txt'
    with open(error_file_path, 'w') as f:
        f.write(f"Total number of error fonts: {len(error_fonts)}\n")
        for font in error_fonts:
            f.write(f"{font}\n")
    print(f"Error fonts saved to {error_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert ttf fonts to sfd fonts")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn', 'vie'], help="Language for charset")
    parser.add_argument("--charset_path", type=str, default='/kaggle/input/font-eng-mini/deepvecfont-v2-minidataset-eng/char_set', help="Path to charset files")
    parser.add_argument("--ttf_path", type=str, default='../data/font_ttfs', help="Path to TTF font files")
    parser.add_argument('--sfd_path', type=str, default='../data/font_sfds', help="Path to save SFD files")
    parser.add_argument('--split', type=str, default='train', help="Data split (train/test)")
    opts = parser.parse_args()
    convert_mp(opts)

if __name__ == "__main__":
    main()
