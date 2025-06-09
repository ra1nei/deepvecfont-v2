import requests
import os
import sys
import json

# --- Cấu hình của bạn ---
# Đảm bảo bạn đã thay thế "YOUR_GOOGLE_FONTS_API_KEY" bằng API Key thực tế của bạn
# (Lưu ý: API Key bạn cung cấp trong prompt có thể không hợp lệ, hãy kiểm tra lại)
API_KEY = "" # Thay thế bằng API Key hợp lệ của bạn
DOWNLOAD_DIR = "downloaded_fonts"
CSS_DIR = "css"

# --- Khai báo các hàm tải xuống và tạo CSS ---
def download_font_file(url, folder, filename):
    """Downloads a font file from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename} from {url}: {e}")
        return False

def generate_css_file(font_family_name, font_styles_data, css_folder):
    """Generates a CSS file for the font family."""
    css_content = ""
    for style, urls_info in font_styles_data.items():
        # Lấy tên file font và đuôi mở rộng từ thông tin đã tải
        file_path_relative = urls_info['local_path_relative_to_css'] # Lấy đường dẫn tương đối đã lưu
        file_ext = file_path_relative.split('.')[-1] # Lấy đuôi mở rộng từ đường dẫn

        css_content += f"@font-face {{\n"
        css_content += f"  font-family: '{font_family_name}';\n"
        # Điều chỉnh font-style và font-weight dựa trên tên style
        css_content += f"  font-style: {'italic' if 'italic' in style.lower() else 'normal'};\n"
        
        # Cố gắng trích xuất font-weight. Ví dụ: 'regular' -> 400, 'bold' -> 700, '500', '600italic'
        weight_str = style.lower().replace('italic', '').strip()
        if not weight_str or weight_str == 'regular':
            font_weight = '400' # Mặc định 400 cho 'regular' hoặc không có weight cụ thể
        elif weight_str == 'bold':
            font_weight = '700'
        else:
            font_weight = weight_str # Nếu là số (ví dụ: '500')
        css_content += f"  font-weight: {font_weight};\n"
        
        # Đường dẫn trong CSS sẽ là relative so với file CSS
        css_content += f"  src: url('../{file_path_relative}') format('{file_ext}');\n"
        css_content += f"}}\n\n"

    css_filepath = os.path.join(css_folder, f"{font_family_name.replace(' ', '_').lower()}.css")
    with open(css_filepath, 'w') as f:
        f.write(css_content)
    print(f"Generated CSS for {font_family_name}: {css_filepath}")


# --- Hàm để lấy tất cả font chỉ hỗ trợ tiếng Việt ---
def get_all_vietnamese_only_fonts_info(api_key):
    """
    Lấy thông tin của tất cả các font từ Google Fonts API
    và lọc ra những font CHỈ hỗ trợ subset 'vietnamese'.
    """
    url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}"
    
    print("Đang lấy danh sách tất cả font từ Google Fonts API...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        all_fonts = data.get("items", [])
        vietnamese_only_fonts = []

        print(f"Đã nhận {len(all_fonts)} họ font. Đang lọc font chỉ hỗ trợ tiếng Việt...")
        for font in all_fonts:
            subsets = font.get("subsets", [])
            # Chỉ chấp nhận font có subset 'vietnamese' VÀ không có 'latin-ext' (nếu bạn muốn loại bỏ hoàn toàn)
            # Hoặc chỉ đơn giản là 'vietnamese' có trong subset
            if "vietnamese" in subsets: # Chỉ kiểm tra 'vietnamese' như yêu cầu mới
                vietnamese_only_fonts.append(font)
                print(f"  > Tìm thấy font tiếng Việt: {font['family']} (Subsets: {', '.join(subsets)})")
        
        return vietnamese_only_fonts

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi lấy danh sách font từ API: {e}")
        return []

# --- Thay đổi phần main execution block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # Lấy thư mục hiện tại của script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Đặt đường dẫn tuyệt đối cho các thư mục download và css
    full_download_dir = os.path.join(SCRIPT_DIR, DOWNLOAD_DIR)
    full_css_dir = os.path.join(SCRIPT_DIR, CSS_DIR)

    # Đảm bảo các thư mục tồn tại
    os.makedirs(full_download_dir, exist_ok=True)
    os.makedirs(full_css_dir, exist_ok=True)

    print("Bắt đầu quá trình tải font tiếng Việt...")
    
    # 1. Lấy danh sách tất cả các font chỉ hỗ trợ tiếng Việt
    # Đổi tên hàm gọi cho rõ ràng hơn
    viet_fonts = get_all_vietnamese_only_fonts_info(API_KEY) 

    if not viet_fonts:
        print("Không tìm thấy font nào hỗ trợ tiếng Việt hoặc có lỗi khi truy cập API. Vui lòng kiểm tra API Key và kết nối mạng.")
    else:
        print(f"\nTổng cộng tìm thấy {len(viet_fonts)} họ font hỗ trợ tiếng Việt. Bắt đầu tải xuống...")
        
        # 2. Lặp qua từng font và tải xuống/tạo CSS
        for font_info in viet_fonts:
            font_family = font_info["family"]
            font_files = font_info["files"] # Đây là một dict chứa các URL cho từng style/weight

            print(f"\n--- Đang xử lý font: {font_family} ---")

            # Tạo thư mục con cho từng họ font bên trong DOWNLOAD_DIR
            current_font_download_dir = os.path.join(full_download_dir, font_family.replace(" ", "_").lower())
            os.makedirs(current_font_download_dir, exist_ok=True)

            downloaded_styles_data = {} # Để lưu trữ thông tin về các file đã tải thành công để tạo CSS

            for style, url_info in font_files.items():
                # url_info có thể là một dict (nếu có nhiều định dạng) hoặc một string (URL trực tiếp của TTF/OTF)
                download_url = None
                file_ext = None

                if isinstance(url_info, dict):
                    # Ưu tiên OTF, sau đó là TTF
                    if 'otf' in url_info:
                        download_url = url_info['otf']
                        file_ext = 'otf'
                    elif 'ttf' in url_info:
                        download_url = url_info['ttf']
                        file_ext = 'ttf'
                    # Các định dạng khác (woff, woff2) bị bỏ qua như yêu cầu
                elif isinstance(url_info, str):
                    # Nếu url_info là một chuỗi, đây thường là URL của định dạng TTF mặc định
                    # Đảm bảo URL này kết thúc bằng .ttf hoặc .otf
                    if url_info.lower().endswith('.otf'):
                        download_url = url_info
                        file_ext = 'otf'
                    elif url_info.lower().endswith('.ttf'):
                        download_url = url_info
                        file_ext = 'ttf'
                    # Nếu nó là woff/woff2 string, bỏ qua
                    
                if download_url:
                    # Tạo tên file theo định dạng chuẩn: fontfamily-style.ext
                    filename = f"{font_family.replace(' ', '-')}-{style}.{file_ext}".lower()
                    
                    # Tải file về thư mục con của font
                    if download_font_file(download_url, current_font_download_dir, filename):
                        # Lưu thông tin đường dẫn tương đối cho CSS
                        downloaded_styles_data[style] = {
                            # Đường dẫn tương đối từ thư mục CSS đến file font
                            'local_path_relative_to_css': os.path.join(DOWNLOAD_DIR, os.path.basename(current_font_download_dir), filename)
                        }
                else:
                    print(f"  Không có URL tải xuống định dạng OTF/TTF cho style '{style}' của font '{font_family}'")

            # 3. Tạo file CSS cho font này sau khi đã tải xuống các style
            if downloaded_styles_data: # Chỉ tạo CSS nếu có ít nhất một style được tải
                generate_css_file(font_family, downloaded_styles_data, full_css_dir)
            else:
                print(f"  Không có style nào của font '{font_family}' được tải xuống (không có OTF/TTF). Bỏ qua tạo CSS.")

        print("\nHoàn thành việc tải xuống tất cả các font tiếng Việt.")