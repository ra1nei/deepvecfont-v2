import fontforge
import os
import multiprocessing as mp
import argparse
import traceback # Để in lỗi chi tiết hơn nếu cần

def worker(task_args):
    """Hàm xử lý cho một font, được gọi bởi Pool."""
    font_idx, font_name, opts, charset, charset_lenw, fonts_dir, sfd_path = task_args
    font_id = os.path.splitext(font_name)[0] # Lấy tên không có phần mở rộng
    split = opts.split
    font_file_path = os.path.join(fonts_dir, font_name)
    target_dir = os.path.join(sfd_path, opts.language, split, font_id) # Thêm language vào đường dẫn đích

    # Sử dụng dict để lưu lỗi chi tiết hơn: {font_name: [list of errors/failed chars]}
    error_details = {}
    failed_chars_list = []

    # 1. Mở font nguồn
    try:
        cur_font = fontforge.open(font_file_path)
    except Exception as e:
        print(f"[{font_idx+1}] ERROR: Không thể mở font {font_name}: {e}")
        error_details[font_name] = [f"Lỗi mở file: {e}"]
        return error_details # Trả về lỗi ngay lập tức

    # 2. Tạo thư mục đích nếu chưa có
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except OSError as e:
            print(f"[{font_idx+1}] ERROR: Không thể tạo thư mục {target_dir}: {e}")
            error_details[font_name] = [f"Lỗi tạo thư mục: {e}"]
            cur_font.close()
            return error_details

    processed_count = 0
    # 3. Lặp qua từng ký tự trong charset
    for char_id, char in enumerate(charset):
        unicode_val = -1 # Khởi tạo để tránh lỗi nếu ord(char) thất bại
        new_font_for_char = None # Khởi tạo để có thể close trong finally
        try:
            unicode_val = ord(char)
            glyph_selector = ('unicode', None), unicode_val # Luôn chọn bằng Unicode

            # --- Chọn và Copy từ nguồn ---
            cur_font.selection.none() # Bỏ chọn tất cả trước
            cur_font.selection.select(glyph_selector)

            # Kiểm tra xem glyph có tồn tại không
            if not cur_font.selection.count:
                # print(f"[{font_idx+1}] Info: Ký tự '{char}' (U+{unicode_val:04X}) không tìm thấy trong {font_name}. Bỏ qua.")
                continue # Bỏ qua ký tự này, không phải lỗi nghiêm trọng

            cur_font.copy()

            # --- Tạo font mới, đặt encoding ---
            new_font_for_char = fontforge.font()
            new_font_for_char.encoding = 'UnicodeBMP' # QUAN TRỌNG

            # --- Chọn slot đích và Paste ---
            new_font_for_char.selection.select(glyph_selector) # Chọn slot tương ứng trong font mới
            new_font_for_char.paste()

            # --- Đặt metadata cho font mới ---
            sfd_base_name = f"{font_id}_{char_id:0{charset_lenw}}"
            new_font_for_char.fontname = sfd_base_name
            new_font_for_char.familyname = font_id # Hoặc tên gốc của font
            new_font_for_char.fullname = sfd_base_name

            # --- Lưu file SFD mới ---
            sfd_filename = os.path.join(target_dir, f'{sfd_base_name}.sfd')
            new_font_for_char.save(sfd_filename)

            # --- Ghi file mô tả (.txt) ---
            try:
                # Truy cập glyph trong font mới bằng unicode_val
                glyph_in_new_font = new_font_for_char[unicode_val]
                desc_filename = os.path.join(target_dir, f'{sfd_base_name}.txt')
                with open(desc_filename, 'w', encoding='utf-8') as char_description:
                    char_description.write(str(unicode_val) + '\n')
                    char_description.write(str(glyph_in_new_font.width) + '\n')
                    char_description.write(str(glyph_in_new_font.vwidth) + '\n')
                    char_description.write(f'{char_id:0{charset_lenw}}\n')
                    char_description.write(f'{font_id}\n')
                    char_description.write(f'{char}\n') # Thêm ký tự gốc cho dễ đọc
                processed_count += 1
            except KeyError:
                 # Lỗi này xảy ra nếu paste thất bại hoặc glyph không được tạo đúng cách
                 print(f"[{font_idx+1}] ERROR: Không thể truy cập glyph U+{unicode_val:04X} trong {sfd_base_name}.sfd sau khi paste.")
                 failed_chars_list.append(f"'{char}' (U+{unicode_val:04X}) - Lỗi truy cập glyph sau paste")
            except Exception as e_write:
                 print(f"[{font_idx+1}] ERROR: Lỗi ghi file txt cho '{char}' (U+{unicode_val:04X}) trong {font_name}: {e_write}")
                 failed_chars_list.append(f"'{char}' (U+{unicode_val:04X}) - Lỗi ghi file txt: {e_write}")

        except Exception as e_char:
            # Lỗi xảy ra khi xử lý ký tự char cụ thể này
            error_msg = f"'{char}' (U+{unicode_val:04X}) - Lỗi xử lý: {e_char}"
            print(f"[{font_idx+1}] ERROR: Xảy ra lỗi với ký tự '{char}' (U+{unicode_val:04X}) trong font {font_name}: {e_char}")
            # print(traceback.format_exc()) # Bỏ comment để xem chi tiết lỗi
            failed_chars_list.append(error_msg)
        finally:
             # Đảm bảo đóng font mới dù có lỗi hay không
             if new_font_for_char:
                 try:
                     new_font_for_char.close()
                 except:
                     pass # Bỏ qua lỗi khi đóng

    # 4. Đóng font nguồn
    cur_font.close()

    # 5. Ghi lại lỗi nếu có
    if failed_chars_list:
        error_details[font_name] = failed_chars_list

    # print(f"[{font_idx+1}] Hoàn thành {font_name}. Xử lý: {processed_count} ký tự. Lỗi ký tự: {len(failed_chars_list)}.")
    return error_details


def convert_mp(opts):
    """Sử dụng multiprocessing Pool để chuyển đổi font."""
    # Kiểm tra charset path
    if not os.path.exists(opts.charset_path):
        print(f"Lỗi: Đường dẫn charset không tồn tại: {opts.charset_path}")
        return
    charset_file_path = os.path.join(opts.charset_path, f"{opts.language}.txt")
    if not os.path.exists(charset_file_path):
        print(f"Lỗi: File charset cho ngôn ngữ '{opts.language}' không tìm thấy tại {charset_file_path}")
        return

    try:
        # Đọc charset với encoding utf-8
        with open(charset_file_path, 'r', encoding='utf-8') as f:
            charset = f.read().strip() # strip() để loại bỏ khoảng trắng thừa
        if not charset:
            print(f"Lỗi: File charset {charset_file_path} rỗng.")
            return
        charset_lenw = len(str(len(charset)))
        print(f"Đã tải {len(charset)} ký tự từ {charset_file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file charset {charset_file_path}: {e}")
        return

    # Kiểm tra ttf path
    if not os.path.exists(opts.ttf_path):
        print(f"Lỗi: Đường dẫn TTF không tồn tại: {opts.ttf_path}")
        return
    fonts_dir = os.path.join(opts.ttf_path, opts.language, opts.split)
    if not os.path.exists(fonts_dir):
        print(f"Lỗi: Thư mục font {fonts_dir} không tồn tại.")
        return

    # Lấy danh sách file font (ttf hoặc otf)
    try:
        ttf_fnames = [f for f in os.listdir(fonts_dir)
                      if os.path.isfile(os.path.join(fonts_dir, f)) and f.lower().endswith(('.ttf', '.otf'))]
    except OSError as e:
        print(f"Lỗi khi truy cập thư mục font {fonts_dir}: {e}")
        return

    if not ttf_fnames:
        print(f"Không tìm thấy file font (.ttf, .otf) nào trong {fonts_dir}")
        return

    font_num = len(ttf_fnames)
    print(f"Tìm thấy {font_num} font để xử lý trong {fonts_dir}.")

    # Tạo thư mục SFD gốc nếu chưa có
    sfd_base_path = os.path.join(opts.sfd_path, opts.language, opts.split)
    if not os.path.exists(sfd_base_path):
        try:
            os.makedirs(sfd_base_path)
            print(f"Đã tạo thư mục đích: {sfd_base_path}")
        except OSError as e:
            print(f"Lỗi: Không thể tạo thư mục đích {sfd_base_path}: {e}")
            return

    # Xác định số tiến trình
    process_num = max(1, mp.cpu_count() - 1) # Để lại 1 core cho hệ thống
    print(f"Sử dụng {process_num} tiến trình.")

    # --- Chuẩn bị tác vụ cho Pool ---
    tasks = []
    for i, fname in enumerate(ttf_fnames):
        # Truyền các tham số cần thiết cho worker
        tasks.append((i, fname, opts, charset, charset_lenw, fonts_dir, opts.sfd_path))

    # --- Chạy Pool ---
    overall_errors = {}
    print("Bắt đầu xử lý font...")
    try:
        pool = mp.Pool(processes=process_num)
        # results là một list các dict lỗi trả về từ mỗi worker
        results = pool.map(worker, tasks)
        pool.close() # Ngăn không cho thêm task mới
        pool.join()  # Đợi tất cả các worker hoàn thành
        print("Hoàn thành xử lý.")
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG trong quá trình đa xử lý: {e}")
        print(traceback.format_exc())
        if 'pool' in locals():
            pool.terminate() # Cố gắng dừng các tiến trình con
            pool.join()
        return # Thoát sớm

    # --- Thu thập và xử lý kết quả lỗi ---
    for res_dict in results:
        if res_dict: # Chỉ cập nhật nếu dict lỗi không rỗng
            overall_errors.update(res_dict)

    # --- Lưu thông tin lỗi ra file ---
    if overall_errors:
        error_file_path = os.path.join(opts.sfd_path, f'errors_{opts.language}_{opts.split}.log') # Lưu gần output
        print(f"\nPhát hiện lỗi trong {len(overall_errors)} font. Chi tiết được lưu tại: {error_file_path}")
        try:
            with open(error_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Báo cáo lỗi xử lý font - Ngôn ngữ: '{opts.language}', Split: '{opts.split}'\n")
                f.write(f"Tổng số font có lỗi: {len(overall_errors)}\n")
                f.write("="*40 + "\n")
                for font, errors in overall_errors.items():
                    f.write(f"Font: {font}\n")
                    for err_detail in errors:
                        f.write(f"  - {err_detail}\n")
                    f.write("-" * 20 + "\n")
        except Exception as e:
            print(f"Lỗi khi ghi file log lỗi {error_file_path}: {e}")
    else:
        print("\nXử lý hoàn tất, không có lỗi nào được báo cáo.")

def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi font ttf sang định dạng glyph riêng lẻ sfd")
    parser.add_argument("--language", type=str, default='vie', choices=['eng', 'chn', 'vie'], help="Ngôn ngữ cho bộ ký tự")
    parser.add_argument("--charset_path", type=str, default='./char_set', help="Đường dẫn đến thư mục chứa file charset (.txt)") # Sửa default path nếu cần
    parser.add_argument("--ttf_path", type=str, default='../data/font_ttfs', help="Đường dẫn gốc đến thư mục chứa font TTF/OTF theo cấu trúc language/split")
    parser.add_argument('--sfd_path', type=str, default='../data/font_sfds', help="Đường dẫn gốc để lưu file SFD output")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help="Phân chia dữ liệu (train/test)")
    opts = parser.parse_args()
    convert_mp(opts)

if __name__ == "__main__":
    # Đặt phương thức start cho multiprocessing (quan trọng trên macOS và Windows)
    # mp.set_start_method('spawn', force=True) # Bỏ comment nếu chạy trên Win/macOS và gặp lỗi fork
    main()