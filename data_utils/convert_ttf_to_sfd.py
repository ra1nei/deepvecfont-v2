import fontforge
import os
import multiprocessing as mp
import argparse
from tqdm import tqdm # Để hiển thị thanh tiến trình đẹp mắt
import sys

# Sử dụng Lock để in an toàn từ nhiều tiến trình
print_lock = mp.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# Hàm này sẽ xử lý một tệp font duy nhất
def process_single_font(task_args):
    """
    Xử lý một tệp font TTF/OTF, trích xuất từng glyph và lưu thành các tệp SFD riêng lẻ.

    Args:
        task_args (tuple): Tuple chứa các đối số:
            - font_file_path (str): Đường dẫn đầy đủ đến tệp font.
            - charset (str): Chuỗi chứa tất cả các ký tự cần xử lý.
            - charset_lenw (int): Chiều rộng định dạng cho ID ký tự.
            - sfd_base_path (str): Đường dẫn thư mục gốc để lưu tệp SFD.
            - language (str): Ngôn ngữ của bộ ký tự.
            - split (str): Loại phân chia dữ liệu (train/test).

    Returns:
        str or None: Tên tệp font nếu có lỗi, ngược lại là None.
    """
    font_file_path, charset, charset_lenw, sfd_base_path, language, split = task_args
    font_name = os.path.basename(font_file_path)
    font_id = font_name.split('.')[0]
    
    if not os.path.exists(font_file_path):
        safe_print(f"❌ File font không tồn tại: {font_file_path}")
        return font_name

    cur_font = None 
    # Mở os.devnull để chuyển hướng stderr
    # 'w' là chế độ ghi, 'encoding' không cần thiết cho devnull nhưng có thể thêm nếu muốn
    # open(os.devnull, 'w') trả về một file object.
    devnull_fd = None # Khởi tạo biến để đóng sau này
    
    try:
        # Chuyển hướng stderr sang os.devnull
        original_stderr = sys.stderr
        devnull_fd = open(os.devnull, 'w')
        sys.stderr = devnull_fd

        cur_font = fontforge.open(font_file_path)
        cur_font.encoding = "UnicodeFull"
    except Exception as e:
        sys.stderr = original_stderr # Khôi phục stderr trước khi in lỗi
        if devnull_fd: # Đảm bảo đóng devnull_fd
            devnull_fd.close()
        safe_print(f"❌ Không thể mở font {font_name} từ '{font_file_path}': {e}")
        if cur_font:
            cur_font.close()
        return font_name
    finally:
        # Quan trọng: Đảm bảo khôi phục stderr và đóng devnull_fd trong mọi trường hợp
        sys.stderr = original_stderr 
        if devnull_fd:
            devnull_fd.close()

    target_dir = os.path.join(sfd_base_path, language, split, f"{font_id}")
    os.makedirs(target_dir, exist_ok=True)

    error_during_char_processing = False

    for char_id, char in enumerate(charset):
        devnull_fd_inner = None # Khởi tạo biến cho vòng lặp trong
        try:
            # Chuyển hướng stderr một lần nữa cho mỗi thao tác FontForge bên trong vòng lặp
            original_stderr_inner = sys.stderr
            devnull_fd_inner = open(os.devnull, 'w')
            sys.stderr = devnull_fd_inner

            new_font_for_char = fontforge.font()
            new_font_for_char.encoding = 'UnicodeFull'

            cur_font.selection.select((ord(char)))
            cur_font.copy()

            new_font_for_char.selection.select((ord(char)))
            new_font_for_char.paste()

            new_font_for_char.fontname = f"{font_id}_{font_name.replace('.', '_')}_{char_id}"
            sfd_file = os.path.join(target_dir, f'{font_id}_{char_id:0{charset_lenw}}.sfd')
            new_font_for_char.save(sfd_file)

            char_file = os.path.join(target_dir, f'{font_id}_{char_id:0{charset_lenw}}.txt')
            with open(char_file, 'w', encoding='utf-8') as char_description:
                char_description.write(f"{char}\n")

                glyph = new_font_for_char[char]

                if glyph is not None and (glyph.width > 0 or glyph.vwidth > 0): 
                    char_description.write(f"{glyph.width}\n")
                    char_description.write(f"{glyph.vwidth}\n")
                else:
                    # Nếu glyph không tồn tại hoặc có width/height là 0 (glyph rỗng/thiếu)
                    safe_print(f"⚠️ Glyph rỗng/thiếu thông tin cho ký tự '{char}' (Unicode: {ord(char)}) trong font '{font_name}'.")
                    char_description.write("0\n0\n")

                char_description.write(f"{char_id:0{charset_lenw}}\n")
                char_description.write(f"{font_id}")

                if char in new_font_for_char:
                    char_description.write(f"{new_font_for_char[char].width}\n")
                    char_description.write(f"{new_font_for_char[char].vwidth}\n")
                else:
                    char_description.write("0\n0\n")

                char_description.write(f"{char_id:0{charset_lenw}}\n")
                char_description.write(f"{font_id}")

            new_font_for_char.close()

        except Exception as e:
            # Chỉ in lỗi nếu đây là lỗi thực sự từ code của bạn, không phải cảnh báo FontForge
            safe_print(f"⚠️ Lỗi khi xử lý glyph '{char}' (Unicode: {ord(char)}) trong font '{font_name}': {e}")
            error_during_char_processing = True
            if new_font_for_char:
                new_font_for_char.close()
        finally:
            sys.stderr = original_stderr_inner # Khôi phục stderr
            if devnull_fd_inner: # Đảm bảo đóng devnull_fd_inner
                devnull_fd_inner.close()


    cur_font.close() 
    
    if error_during_char_processing:
        return font_name
    return None


def convert_fonts_to_sfd(opts):
    """
    Hàm chính để chuyển đổi các tệp font TTF/OTF sang định dạng SFD sử dụng đa xử lý.
    """
    charset_file_path = os.path.join(opts.charset_path, f"{opts.language}.txt")

    if not os.path.exists(charset_file_path):
        safe_print(f"Charset file for language '{opts.language}' not found at {charset_file_path}")
        return

    with open(charset_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charset = ''.join([line.strip() for line in lines if line.strip()])
    charset_lenw = len(str(len(charset))) # Độ dài của ID ký tự (để định dạng tên tệp)

    fonts_base_dir = os.path.join(opts.ttf_path, opts.language, opts.split)
    sfd_base_path = opts.sfd_path

    ttf_file_paths = []
    # Đi bộ qua thư mục để tìm tất cả các tệp font
    for root, dirs, files in os.walk(fonts_base_dir):
        for f in files:
            if f.lower().endswith(('.ttf', '.otf')): # Chỉ xử lý tệp TTF hoặc OTF
                ttf_file_paths.append(os.path.join(root, f))

    if not ttf_file_paths:
        safe_print(f"Không tìm thấy tệp font (.ttf/.otf) trong '{fonts_base_dir}'. Vui lòng kiểm tra đường dẫn.")
        return

    # Xác định số lượng tiến trình
    num_processes = opts.num_processes if opts.num_processes is not None else max(mp.cpu_count() - 2, 1)
    if num_processes <= 0: # Đảm bảo số tiến trình ít nhất là 1
        num_processes = 1

    # Chuẩn bị danh sách các nhiệm vụ (tasks) cho Pool
    # Mỗi nhiệm vụ là một tuple chứa các đối số cho hàm process_single_font
    tasks = [(font_path, charset, charset_lenw, sfd_base_path, opts.language, opts.split)
             for font_path in ttf_file_paths]

    # Xác định chunksize
    # Nếu font_per_chunk được chỉ định, sử dụng giá trị đó.
    # Ngược lại, tính toán một chunksize mặc định hợp lý.
    # chunksize = 1 thường an toàn nhất nếu FontForge rất nặng về tài nguyên,
    # nhưng có thể tăng overhead. Giá trị lớn hơn giúp giảm overhead.
    calculated_chunksize = opts.font_per_chunk
    if calculated_chunksize is None and len(tasks) > 0:
        # Một heuristic phổ biến: tổng số nhiệm vụ / (số lượng tiến trình * N)
        # N thường là 4, nhưng có thể điều chỉnh tùy theo độ nặng của từng nhiệm vụ.
        # Ở đây, mình dùng 1, nếu quá chậm có thể tăng lên.
        calculated_chunksize = max(1, len(tasks) // (num_processes * 4)) # Đảm bảo ít nhất là 1
        if calculated_chunksize == 0 and len(tasks) > 0: # Tránh chunksize = 0 nếu có nhiệm vụ
            calculated_chunksize = 1

    safe_print(f"\n--- Bắt đầu chuyển đổi Fonts ---")
    safe_print(f"Tổng số font tìm thấy: {len(ttf_file_paths)}")
    safe_print(f"Số lượng tiến trình sử dụng: {num_processes}")
    safe_print(f"Kích thước chunk (font mỗi tiến trình nhận một lần): {calculated_chunksize}")
    safe_print(f"Lưu SFD tại: {sfd_base_path}\n")

    error_fonts = []
    # Sử dụng multiprocessing.Pool để phân phối công việc
    with mp.Pool(processes=num_processes) as pool:
        # imap_unordered cho phép kết quả trả về không theo thứ tự, 
        # rất tốt để hiển thị tiến trình và xử lý các lỗi ngay khi chúng xảy ra.
        # tqdm bọc ngoài để hiển thị thanh tiến trình.
        for result in tqdm(pool.imap_unordered(process_single_font, tasks, chunksize=calculated_chunksize),
                           total=len(tasks), desc="Đang xử lý Fonts"):
            if result is not None: # Nếu hàm process_single_font trả về tên font, đó là lỗi
                error_fonts.append(result)

    # Ghi danh sách các font bị lỗi vào tệp
    error_file_path = os.path.join(os.getcwd(), 'error_fonts.txt') # Lưu trong thư mục làm việc hiện tại
    with open(error_file_path, 'w', encoding='utf-8') as f:
        unique_error_fonts = sorted(list(set(error_fonts))) # Lọc trùng lặp và sắp xếp
        f.write(f"Tổng số font gặp lỗi: {len(unique_error_fonts)}\n")
        if unique_error_fonts:
            f.write("Danh sách các font bị lỗi:\n")
            for font in unique_error_fonts:
                f.write(f"- {font}\n")
        else:
            f.write("Không có font nào gặp lỗi trong quá trình xử lý.\n")

    safe_print(f"\n✅ Đã hoàn tất quá trình xử lý.")
    safe_print(f"Tóm tắt lỗi được lưu vào: {error_file_path}")
    safe_print(f"Tổng số font gặp lỗi: {len(unique_error_fonts)}")

def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi các font TTF/OTF sang định dạng SFD.")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn', 'vie'],
                        help="Ngôn ngữ của bộ ký tự. ('eng', 'chn', 'vie')")
    parser.add_argument("--charset_path", type=str, default='../data/char_set',
                        help="Đường dẫn đến thư mục chứa các tệp bộ ký tự (ví dụ: eng.txt).")
    parser.add_argument("--ttf_path", type=str, default='../data/font_ttfs',
                        help="Đường dẫn đến thư mục gốc chứa các tệp font TTF/OTF.")
    parser.add_argument('--sfd_path', type=str, default='../data/font_sfds',
                        help="Đường dẫn đến thư mục để lưu các tệp SFD đã chuyển đổi.")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'val'],
                        help="Tên thư mục con chứa font (ví dụ: 'train', 'test', 'val').")
    parser.add_argument('--num_processes', type=int, default=None,
                        help="Số lượng tiến trình (processes) để sử dụng. Mặc định là số lượng CPU - 2.")
    parser.add_argument('--font_per_chunk', type=int, default=None,
                        help="Số lượng font mà mỗi tiến trình nhận và xử lý tại một thời điểm (chunksize). "
                             "Nếu không được chỉ định, sẽ tự động tính toán.")

    opts = parser.parse_args()
    convert_fonts_to_sfd(opts)

if __name__ == "__main__":
    main()