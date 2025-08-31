import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import json
import os
import tempfile
import hashlib
import pytesseract
from PIL import Image
import io
import fitz
import google.generativeai as genai
from openai import OpenAI
from openai import OpenAIError
from docx import Document
from dotenv import load_dotenv
import base64
import pandas as pd
from pathlib import Path
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


# Load biến môi trường từ file .env
load_dotenv()

# Lấy API key từ biến môi trường
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Nếu không có cả Google API key và OpenAI API key thì báo lỗi và dừng streamlit
if not google_api_key and not openai_api_key:
    st.error("Không tìm thấy API key nào (Google hoặc OpenAI)")
    st.stop()

# Nếu có Google API key thì cấu hình cho thư viện google.generativeai
if google_api_key:
    genai.configure(api_key=google_api_key)
    

# Đọc file JSON
def load_language_strings(file_path="language_strings.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Không thể đọc file language_strings.json: {e}")
        return {}  # Trả về từ điển rỗng nếu lỗi

# Tải từ điển ngôn ngữ
LANGUAGE_STRINGS = load_language_strings()
if not LANGUAGE_STRINGS:
    st.stop()  # Dừng ứng dụng nếu không đọc được file JSON

# Khởi tạo trạng thái ngôn ngữ
if "language" not in st.session_state:
    st.session_state["language"] = "vi"  # Ngôn ngữ mặc định là tiếng Việt
    
def save_bytes_to_tempfile(file_bytes, suffix):
    """Ghi bytes vào file tạm và trả về đường dẫn."""
    # Tạo 1 file tạm (temporary file) với phần đuôi (suffix) do người dùng truyền vào
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # Ghi dữ liệu dạng bytes vào file tạm
    tmp.write(file_bytes)
    # Đảm bảo dữ liệu đã được ghi hết xuống ổ đĩa
    tmp.flush()
    # Đóng file
    tmp.close()
    # Trả về đường dẫn file tạm để có thể mở lại sau
    return tmp.name

def image_bytes_ocr(img_bytes):
    try:
        # Chuyển bytes thành đối tượng ảnh bằng Pillow
        img = Image.open(io.BytesIO(img_bytes))
        # Dùng pytesseract OCR nhận diện chữ (cả tiếng Anh + tiếng Việt)
        text = pytesseract.image_to_string(img, lang='eng+vie')
        # Loại bỏ khoảng trắng đầu/cuối và trả về
        return text.strip()
    except Exception as e:
        # Nếu có lỗi (ảnh hỏng, OCR lỗi...) thì trả về chuỗi rỗng
        return ""

def clear_chat_history():
    # Xóa lịch sử chat trong session state của Streamlit.
    st.session_state["chat_history"] = []

def get_text_from_file(file_bytes, filename):
    # Lấy định dạng file
    ext = Path(filename).suffix.lower()
    tmp_path = None
    try:
        # ==================Xử lý PDF ==================
        if ext in [".pdf"]:
            # Lưu file PDF ra file tạm
            tmp_path = save_bytes_to_tempfile(file_bytes, suffix=".pdf")
            
            # Dùng PyPDFLoader để đọc văn bản trong PDF, chia thành nhiều trang
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
            
            # Ghép nội dung các trang thành 1 chuỗi text
            text = "\n".join(p.page_content for p in pages)
            
            # Preview mặc định cho PDF (sẽ hiển thị nhúng PDF sau)
            preview = None  
            images = []  # chứa các ảnh trích xuất từ PDF
            ocr_texts = [] # chứa văn bản OCR được trích xuất từ ảnh
            seen_hashes = set() # để tránh xử lý trùng ảnh
            
            # Mở PDF bằng PyMuPDF (fitz) để trích xuất ảnh
            pdf_doc = fitz.open(tmp_path)
            for page in pdf_doc:
                for img in page.get_images(full=True):
                    xref = img[0] # ID ảnh
                    base_image = pdf_doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_format = base_image["ext"]
                    
                    # Tính hash để tránh duplicate ảnh
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    if img_hash not in seen_hashes:
                        seen_hashes.add(img_hash)
                        
                        # Chuyển ảnh thành base64 để hiển thị trong Streamlit
                        b64_img = base64.b64encode(img_bytes).decode("utf-8")
                        images.append({"format": img_format, "b64": b64_img})
                        
                        # OCR văn bản trong ảnh 
                        ocr_txt = image_bytes_ocr(img_bytes)
                        if ocr_txt:
                            ocr_texts.append(ocr_txt)
            # Nếu có văn bản OCR từ ảnh thì nối thêm vào text
            if ocr_texts:
                text += "\n\n[Text trích xuất từ ảnh]\n" + "\n".join(ocr_texts)
            # Tạo preview cho PDF (hiển thị text + ảnh trích xuất)
            preview = {"type": "pdf", "text_preview": text[:1000], "images": images}
            return text, preview

        # ==================== XỬ LÝ DOCX ====================
        elif ext in [".docx"]:
            # Lưu file DOCX ra file tạm
            tmp_path = save_bytes_to_tempfile(file_bytes, suffix=".docx")
            
            # Dùng Unstructured loader để đọc văn bản
            loader = UnstructuredWordDocumentLoader(tmp_path)
            pages = loader.load_and_split()
            text = "\n".join(p.page_content for p in pages)
            
            # Đọc file docx bằng python-docx để trích xuất ảnh
            doc = Document(tmp_path)
            images = []
            ocr_texts = []
            seen_hashes = set()
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_bytes = rel.target_part.blob
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    if img_hash not in seen_hashes:
                        seen_hashes.add(img_hash)
                        
                        # Lưu ảnh dạng base64
                        b64_img = base64.b64encode(img_bytes).decode("utf-8")
                        images.append({"format": "png", "b64": b64_img})
                        
                        # OCR text trong ảnh (nếu có)
                        ocr_txt = image_bytes_ocr(img_bytes)
                        if ocr_txt:
                            ocr_texts.append(ocr_txt)
                            
            # Nếu OCR được text từ ảnh thì nối vào văn bản
            if ocr_texts:
                text += "\n\n[Text trích xuất từ ảnh]\n" + "\n".join(ocr_texts)
            # Tạo preview cho DOCX
            preview = {"type": "docx", "text_preview": text[:30000000], "images": images}
            return text, preview
        
        # ==================== XỬ LÝ EXCEL & CSV ====================
        elif ext in [".xlsx", ".csv"]:
            tmp_path = save_bytes_to_tempfile(file_bytes, suffix=ext)
            
            # Đọc dữ liệu bảng
            if ext == ".csv":
                df = pd.read_csv(tmp_path)
                sheets = {"Sheet1": df} # CSV coi như 1 sheet
            else:
                # Đọc tất cả sheet trong Excel
                sheets = pd.read_excel(tmp_path, sheet_name=None, engine="openpyxl")
            text_parts = []
            preview = {"type": "xlsx", "sheets": sheets}
            
            # Convert dữ liệu mỗi sheet sang dạng text (giống CSV)
            for sheet_name, df in sheets.items():
                csv_like = df.to_csv(index=False)
                text_parts.append(f"Sheet: {sheet_name}\n{csv_like}")
                
            # Ghép toàn bộ sheet thành text
            text = "\n\n".join(text_parts)
            return text, preview
        # ==================== ĐỊNH DẠNG KHÔNG HỖ TRỢ ====================
        else:
            return "", {"type": "unknown", "msg": f"Định dạng {ext} chưa được hỗ trợ"}
    
    except Exception as e:
        # Báo lỗi ra Streamlit nếu có sự cố khi đọc file
        st.error(f"Lỗi đọc file {filename}: {e}")
        return "", {"type": "error", "msg": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def make_df_arrow_compatible(df):
    """Chuyển đổi DataFrame để tương thích với Arrow serialization"""
    for col in df.columns:
        dtype = df[col].dtype
        
        # Nếu cột dùng pandas extension dtypes (vd: Int64, StringDtype, BooleanDtype)
        if pd.api.types.is_extension_array_dtype(dtype):
            
            # Nếu là số nguyên (nullable int)
            if pd.api.types.is_integer_dtype(dtype):
                if df[col].isna().any():
                    # Nếu có NaN thì không thể giữ int => chuyển thành float64
                    df[col] = df[col].astype('float64')
                else:
                    # Nếu không có NaN thì giữ int64 chuẩn
                    df[col] = df[col].astype('int64')
            
            # Nếu là số thực (float dtype nhưng ở dạng extension)
            elif pd.api.types.is_float_dtype(dtype):
                df[col] = df[col].astype('float64')
            
            # Nếu là chuỗi (StringDtype)
            elif pd.api.types.is_string_dtype(dtype):
                # Ép về object (chuẩn hơn cho Arrow)
                df[col] = df[col].astype('object')
            
            # Các loại extension khác (vd: category, boolean nullable)
            else:
                # Chuyển về object để đảm bảo không lỗi
                df[col] = df[col].astype('object')
        
        # Nếu là object dtype (có thể chứa nhiều loại dữ liệu)
        if dtype == 'object':
            pass
    return df

def show_preview_from_file(file_bytes, filename, preview_info):
    lang = st.session_state["language"]
    if preview_info is None:
        st.info(LANGUAGE_STRINGS[lang]["no_preview"])
        return
    
    # ====== Preview cho PDF ======
    if preview_info.get("type") == "pdf":
        st.subheader(LANGUAGE_STRINGS[lang]["pdf_preview"].format(filename=filename))
        
        # Mã hóa PDF thành base64 rồi nhúng vào iframe để hiển thị trong Streamlit
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Hiển thị text trích xuất từ PDF (preview ngắn)
        st.text(preview_info.get("text_preview", ""))
        
        # Nếu PDF có chứa ảnh thì hiển thị thêm
        images = preview_info.get("images", [])
        if images:
            st.write("Ảnh trong file:" if lang == "vi" else "Images in file:")
            for img in images:
                st.image(f"data:image/{img['format']};base64,{img['b64']}", width="stretch")
    
    # ====== Preview cho DOCX ======
    elif preview_info.get("type") == "docx":
        st.subheader(f"Preview - {filename} (docx)")
        st.text(preview_info.get("text_preview", ""))
        
        # Hiển thị ảnh trong file Word (nếu có)
        images = preview_info.get("images", [])  
        if images:
            st.write("Ảnh trong file:" if lang == "vi" else "Images in file:")
            for img in images:
                st.image(f"data:image/{img['format']};base64,{img['b64']}", width="stretch")
    
    # ====== Preview cho Excel/CSV ======
    elif preview_info.get("type") == "xlsx":
        st.subheader(LANGUAGE_STRINGS[lang]["xlsx_preview"].format(filename=filename))
        sheets = preview_info["sheets"]
        
        # Cho phép chọn sheet để hiển thị
        selected_sheet = st.selectbox(
            LANGUAGE_STRINGS[lang]["sheet_select"].format(filename=filename),
            list(sheets.keys()),
            key=f"select_sheet_{hash(filename)}"
        )
        if selected_sheet:
            df = sheets[selected_sheet]
            df = make_df_arrow_compatible(df)  # Sửa lỗi Arrow serialization
            st.write(f"Sheet: {selected_sheet}")
            st.dataframe(df.head(10))

            if df is not None:
                # Đưa ra bảng thống kê mô tả nhanh (tự động tính count, mean, std, min, 25%, 50%, 75%, max)
                st.subheader(LANGUAGE_STRINGS[lang]["stats_header"])
                st.write(df.describe())
                
                # Hiển thị số dòng và số cột của data
                st.subheader(LANGUAGE_STRINGS[lang]["shape_header"])
                st.write(df.shape)
                
                # Hiển thị bảng các loại kiểu dữ liệu của từng cột
                st.subheader(LANGUAGE_STRINGS[lang]["dtypes_header"])
                st.write(df.dtypes.astype(str))  
                
                # Hiển thị số lượng null data
                st.subheader(LANGUAGE_STRINGS[lang]["null_header"])
                st.write(df.isnull().sum())
                
                # Hiển thị số dòng bị trùng lặp
                st.subheader(LANGUAGE_STRINGS[lang]["duplicates_header"])
                st.write(df.duplicated().sum())
            
            st.header(LANGUAGE_STRINGS[lang]["data_handling_header"])
            # Đổi kiểu dữ liệu nếu đang ở dạng object (TH đặc biệt nếu là ngày tháng năm thì đổi thành datetime64[ns]. Còn lại thì chuyển thành kiểu dữ liệu phù hợp)
            st.subheader(LANGUAGE_STRINGS[lang]["dtype_change_header"])
            for col in df.columns:
                current_dtype = df[col].dtype
                if current_dtype == "object" :
                    try:
                        df[col] = pd.to_datetime(df[col], errors="raise", dayfirst=True)
                    except Exception:
                        pass
            st.write(df.dtypes.astype(str))  
            
             # --- Xử lý dữ liệu trùng lặp ---
            st.subheader(LANGUAGE_STRINGS[lang]["duplicates_handling_header"])
            # Tạo ra sự lựa chọn không làm hoặc xử lý dữ liệu bị trùng
            cleaned_duplicated_data_actions = {
                LANGUAGE_STRINGS[lang]["duplicates_options"]["do_not_do_anything"]: "do_not_do_anything",
                LANGUAGE_STRINGS[lang]["duplicates_options"]["dropna_the_duplicated_rows"]: "dropna_the_duplicated_rows"
            }
            # Sử dụng bảng băm (hash) để tạo một giá trị duy nhất (unique key) cho mỗi filename 
            key_id = hash(filename + selected_sheet)
            selected_label = st.radio(
                "Duplicated Handling Strategy", 
                list(cleaned_duplicated_data_actions.keys()), 
                label_visibility="collapsed",
                key=f"duplicated_strategy_{key_id}"
            )
            
            cleaned_duplicated_data_action = cleaned_duplicated_data_actions[selected_label]
            # Không xử lý dữ liệu trùng
            if cleaned_duplicated_data_action == "do_not_do_anything":
                df = df 
            # Xử lý dữ liệu bị trùng bằng cách xóa dòng trùng
            elif cleaned_duplicated_data_action == "dropna_the_duplicated_rows":
                df = df.drop_duplicates()
            
            
            if df is not None: 
            # Hiển thị bảng dataframe sau khi xử lý dữ liệu bị trùng 
                df = make_df_arrow_compatible(df)  # Fix Arrow issues
                st.subheader("📊 Dữ liệu sau khi xử lý trùng lặp:" if lang == "vi" else "📊 The handled duplicated data: ")
                st.dataframe(df)
            # Hiển thị số lượng dòng sau khi dọn sạch
                st.subheader(LANGUAGE_STRINGS[lang]["duplicates_header"] + (" sau khi xử lý" if lang == "vi" else " after cleaned"))
                st.write(df.duplicated().sum())
            
            st.subheader(LANGUAGE_STRINGS[lang]["null_handling_header"])
            # Tạo ra các lựa chọn để xử lý dữ liệu bị rỗng (null)
            cleaned_null_data_actions = {
                LANGUAGE_STRINGS[lang]["null_options"]["none"]: "none",
                LANGUAGE_STRINGS[lang]["null_options"]["dropna"]: "dropna",
                LANGUAGE_STRINGS[lang]["null_options"]["unknown"]: "unknown",
                LANGUAGE_STRINGS[lang]["null_options"]["ffill"]: "ffill",
                LANGUAGE_STRINGS[lang]["null_options"]["bfill"]: "bfill"
            }
            # Sử dụng bảng băm (hash) để tạo một giá trị duy nhất (unique key) cho mỗi filename 
            key_id = hash(filename + selected_sheet)
            selected_label = st.radio(
                "Null Handling Strategy",
                list(cleaned_null_data_actions.keys()),
                label_visibility="collapsed",
                key=f"null_strategy_{key_id}"
            )
            clean_null_data_action = cleaned_null_data_actions[selected_label]
            df_cleaned = df.copy()
            
            # Không xử lý dữ liệu bị rỗng
            if clean_null_data_action == "none":
                df_cleaned = df_cleaned
            
            # Xử lý dữ liệu bị rỗng bằng cách xóa dòng bị null
            elif clean_null_data_action == "dropna":
                df_cleaned = df_cleaned.dropna()
            
            # Xử lý dữ liệu bị rỗng bằng cách thay bằng unknown (nếu là object, string, datetime64[ns]) hoặc 0 (nếu dữ liệu là kiểu có số nguyên hoặc số thực
            elif clean_null_data_action == "unknown":
                object_cols_string = df_cleaned.select_dtypes(include=["object", "string", "datetime64[ns]"]).columns
                object_cols_int_or_float = df_cleaned.select_dtypes(include=["int", "float", "int64", "float64", "Int64", "Float64"]).columns.to_list()
                existing_object_cols = [col for col in object_cols_string if col in df_cleaned.columns]
                if not object_cols_string.empty:
                    df_cleaned[existing_object_cols] = df_cleaned[existing_object_cols].fillna("Unknown")
                    st.info(f"Replaced nulls with 'Unknown' in columns: {', '.join(existing_object_cols)}")
                if object_cols_int_or_float:
                    for col in object_cols_int_or_float:
                        try:
                            df_cleaned[col] = df_cleaned[col].fillna(0)
                        except Exception as e:
                            st.warning(f"⚠️ Could not fill nulls in column `{col}`: {e}")
                    st.info(f"Replaced nulls with 0 in numeric columns: {', '.join(object_cols_int_or_float)}")
            
            # Xử lý dữ liệu bị null bằng sử dụng phương pháp ffill 
            # Phương pháp ffill là forward fill tức là điền giá trị bị thiếu (NaN) bằng giá trị gần nhất phía trước nó (trên cùng một cột).
            elif clean_null_data_action == "ffill":
                df_cleaned = df_cleaned.fillna(method="ffill")
            
            # Xử lý dữ liệu bị null bằng sử dụng phương pháp bfill 
            # Phương pháp bfill là backward fill tức là điền giá trị bị thiếu (NaN) bằng giá trị gần nhất phía sau nó (trên cùng một cột).
            elif clean_null_data_action == "bfill":
                df_cleaned = df_cleaned.fillna(method="bfill")
            
            
            if df_cleaned is not None: 
                # Hiển thị bảng dataframe sau khi xử lý dữ liệu bị rỗng
                df_cleaned = make_df_arrow_compatible(df_cleaned) 
                st.subheader("📊 Dữ liệu sau khi xử lý null:" if lang == "vi" else "📊 The handled null data:")
                st.dataframe(df_cleaned)
                # Hiển thị số lượng dòng sau khi dọn sạch dữ liệu rỗng
                st.subheader(LANGUAGE_STRINGS[lang]["null_header"] + (" sau khi xử lý" if lang == "vi" else " after cleaned "))
                st.write(df_cleaned.isnull().sum())
                
            # Tạo nút download csv sau khi xử lý dữ liệu
            st.subheader(LANGUAGE_STRINGS[lang]["download_button"])
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                LANGUAGE_STRINGS[lang]["download_button"],
                csv,
                file_name=LANGUAGE_STRINGS[lang]["download_filename"].format(filename=filename, sheet=selected_sheet),
                mime="text/csv",
                key=f"download_{key_id}"
            )
            st.subheader(LANGUAGE_STRINGS[lang]["chart_header"])
            
            # Xác định các loại cột trong dataframe
            categorical_cols = df_cleaned.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            numerical_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns.tolist()
            datetime_cols = df_cleaned.select_dtypes(include=["datetime64"]).columns.tolist()
            
            # Select box để chọn loại biểu đồ
            chart_type = st.selectbox(
                LANGUAGE_STRINGS[lang]["chart_type_label"],
                LANGUAGE_STRINGS[lang]["chart_type_options"],
                key=f"chart_type_{key_id}"
            )
            
            # Biểu đồ tròn
            if chart_type == "pie":
                if categorical_cols:
                    selected_cat = st.selectbox(
                        LANGUAGE_STRINGS[lang]["pie_col_label"],
                        categorical_cols,
                        key=f"pie_cat_{key_id}"
                    )
                    value_counts = df_cleaned[selected_cat].value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                    ax.set_title(f"Phân bố của '{selected_cat}' (Top 10)" if lang == "vi" else f"The distribution of '{selected_cat}' (Top 10)")
                    st.pyplot(fig)
                else:
                    st.info("Don't have column to draw the pie chart")
            
            # Biểu đồ cột
            elif chart_type == "bar":
                if categorical_cols and numerical_cols:
                    selected_cat = st.selectbox(
                        LANGUAGE_STRINGS[lang]["bar_cat_label"],
                        categorical_cols,
                        key=f"bar_cat_{key_id}"
                    )
                    selected_num = st.selectbox(
                        LANGUAGE_STRINGS[lang]["bar_num_label"],
                        numerical_cols,
                        key=f"bar_num_{key_id}"
                    )
                    agg_func = st.selectbox(
                        LANGUAGE_STRINGS[lang]["bar_agg_label"],
                        ["mean", "sum", "count", "min", "max"],
                        key=f"bar_agg_{key_id}"
                    )
                    agg_df = df_cleaned.groupby(selected_cat)[selected_num].agg(agg_func).sort_values(ascending=False).reset_index()
            
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x=selected_cat, y=selected_num, data=agg_df, ax=ax, palette="pastel")
                    for index, row in agg_df.iterrows():
                        ax.text(index, row[selected_num], f"{row[selected_num]:.2f}", ha='center', va='bottom', fontsize=8)
                    ax.set_title(f"{agg_func.capitalize()} của {selected_num} theo {selected_cat}" if lang == "vi" else f"{agg_func.capitalize()} of {selected_num} by {selected_cat}")
                    ax.set_xlabel(selected_cat)
                    ax.set_ylabel(f"{agg_func.capitalize()} {selected_num}")
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                else:
                    st.info(LANGUAGE_STRINGS[lang]["no_bar_cols"])
            
            # Biểu đồ đường
            elif chart_type == "line":
                x_cols = numerical_cols + datetime_cols  # Trục x có thể là số hoặc ngày giờ
                if x_cols and numerical_cols:
                    selected_x = st.selectbox(
                        LANGUAGE_STRINGS[lang]["line_x_label"],
                        x_cols,
                        key=f"line_x_{key_id}"
                    )
                    selected_y = st.selectbox(
                        LANGUAGE_STRINGS[lang]["line_y_label"],
                        numerical_cols,
                        key=f"line_y_{key_id}"
                    )
                    if selected_x and selected_y:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sorted_df = df_cleaned.sort_values(by=selected_x)  # Sắp xếp theo trục x
                        ax.plot(sorted_df[selected_x], sorted_df[selected_y], marker='o', linestyle='-', color='b')
                        ax.set_title(f"{selected_y} theo {selected_x}" if lang == "vi" else f"{selected_y} by {selected_x}" )
                        ax.set_xlabel(selected_x)
                        ax.set_ylabel(selected_y)
                        if pd.api.types.is_datetime64_any_dtype(df_cleaned[selected_x]):  # Xử lý trục x nếu là ngày giờ
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                            plt.xticks(rotation=45)
                        ax.grid(True)
                        st.pyplot(fig)
                    else:
                        st.info("Vui lòng chọn cột cho trục ngang và trục dọc" if lang == "vi" else "Please choose column for horizontal and vertical" )
                else:
                    st.info(LANGUAGE_STRINGS[lang]["no_line_cols"])
            
def get_text_chunks(text, chunk_size, chunk_overlap):
    try:
        # Dùng TokenTextSplitter để cắt văn bản thành nhiều phần nhỏ
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception as e:
        # Báo lỗi nếu quá trình chia chunk thất bại
        st.error(f"Lỗi chia chunk: {e}" if lang == "vi" else f"Error splitting chunks: {e}")
        return []

def get_vector_store(text_chunks,session_dir, provider="Gemini"):
    try:
         # Chọn model embedding theo provider
        if provider == "Gemini":
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:  
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
         # Tạo FAISS vector database từ các text_chunks
        vector_store = FAISS.from_texts(text_chunks, embeddings)
         # Lưu vector database vào thư mục session_dir
        vector_store.save_local(session_dir)
        # Thông báo thành công
        st.success("Tài liệu đã được phân tích xong, sẵn sàng trả lời" if lang == "vi" else "Document analysis is done, ready to answer")
    except OpenAIError as e:
        if e.code == "insufficient_quota":
            st.error("Đã vượt quá quota OpenAI. Vui lòng kiểm tra gói dịch vụ hoặc chuyển sang Gemini." if lang == "vi" else "OpenAI quota exceeded. Please check your service plan or switch to Gemini.")
        else:
            st.error(f"Lỗi lưu vector database: {e}" if lang == "vi" else f"Error saving vector database: {e}")
    except Exception as e:
        # Báo lỗi nếu quá trình tạo/lưu vector store thất bại
        st.error(f"Lỗi lưu vector database: {e}" if lang == "vi" else f"Error saving vector database: {e}")

def get_conversational_chain(answer_mode="Chi tiết", show_reasoning=False, provider="Gemini"):
    # Xác định phong cách trả lời
    style = "ngắn gọn, súc tích" if answer_mode == "Ngắn gọn" else "chi tiết, đầy đủ" 

    # Nếu người dùng muốn thấy reasoning steps thì thêm yêu cầu vào prompt
    reasoning_part = ""
    if show_reasoning:
        reasoning_part = "\nNgoài ra, hãy đưa ra một đoạn 'Tóm tắt reasoning steps' giải thích ngắn gọn cách bạn suy luận từ ngữ cảnh.\n"
    
    # Prompt template: ràng buộc model trả lời CHÍNH XÁC theo ngữ cảnh
    prompt_template = f"""
Bạn là một trợ lý AI, nhiệm vụ là trả lời dựa *chính xác* vào ngữ cảnh cung cấp. 
Tuyệt đối **không bịa** nếu ngữ cảnh không có thông tin. 
Nếu không tìm thấy câu trả lời, hãy trả lời đúng nguyên văn: "Câu trả lời không có trong ngữ cảnh."

Yêu cầu: Trả lời theo phong cách {style}.
{reasoning_part}

Ngữ cảnh: {{context}}
Câu hỏi: {{question}}

Answer:
"""

    try:
         # Chọn model theo provider
        if provider == "Gemini": #Gemini
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        else:  # ChatGPT
            model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.1)
         # Gắn prompt template vào chain QA
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Lỗi tạo chain: {e}")
        return None

def user_input(user_question, session_dir, answer_mode, show_reasoning, provider="Gemini"):
    try:
         # Chọn embeddings phù hợp
        if provider == "Gemini":
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Kiểm tra FAISS DB tồn tại chưa
        if not os.path.exists(session_dir):
            st.error("Không tìm thấy FAISS index. Hãy phân tích tài liệu trước." if lang == "vi" else "FAISS index not found. Please analyze document first.")
            error  = "Không tìm thấy dữ liệu để trả lời." if lang == "vi" else "No data found to answer."
            return error

         # Load lại FAISS DB
        new_db = FAISS.load_local(session_dir, embeddings, allow_dangerous_deserialization=True)
        
        # Tìm kiếm các đoạn văn bản gần giống câu hỏi
        docs = new_db.similarity_search(user_question)
        
        # Tạo chain QA
        chain = get_conversational_chain(answer_mode, show_reasoning, provider)
        if not chain:
            error = "Không tạo được chain." if lang == "vi" else "Failed to create chain."
            return error

         # Gọi chain để sinh câu trả lời
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        # Lưu lịch sử hội thoại
        st.session_state["chat_history"].append({
            "question": user_question,
            "answer": answer
        })
        return answer
    except Exception as e:
        error = f"Lỗi xử lý câu hỏi: {e}" if lang == "vi" else f"Error processing question: {e}"
        st.error(error)
        return error

# ---------- Streamlit UI ----------
# Cấu hình app Streamlit
st.set_page_config(page_title="Chatbot RAG")

# Khởi tạo trạng thái ngôn ngữ
if "language" not in st.session_state:
    st.session_state["language"] = "vi"  # Ngôn ngữ mặc định là tiếng Việt
    
# Tạo session_id cho mỗi lần chạy (để lưu vectorstore riêng biệt)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_dir = f"faiss_index_{st.session_state['session_id']}"

# Khởi tạo lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Cấu hình app Streamlit
lang = st.session_state["language"]
st.set_page_config(page_title=LANGUAGE_STRINGS[lang]["title"])
st.title(LANGUAGE_STRINGS[lang]["title"])
# Thanh sidebar: upload file + cấu hình
with st.sidebar:
    st.title(LANGUAGE_STRINGS[lang]["menu"])
    st.selectbox(
        "Ngôn ngữ / Language",
        options=["vi", "en"],
        format_func=lambda x: "Tiếng Việt" if x == "vi" else "English",
        key="language"
    )
    uploaded_files = st.file_uploader(
        LANGUAGE_STRINGS[lang]["upload_label"], 
        accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "csv"]
    )
    
    st.markdown(LANGUAGE_STRINGS[lang]["chunk_size_label"])
    chunk_size = st.number_input(
        LANGUAGE_STRINGS[lang]["chunk_size_label"], 
        min_value=500, 
        max_value=20000, 
        value=10000, 
        step=500, 
        key="chunk_size"
    )
    chunk_overlap = st.number_input(
        LANGUAGE_STRINGS[lang]["chunk_overlap_label"],
        min_value=0,
        max_value=5000,
        value=1000,
        step=100,
        key="chunk_overlap"
    )
    answer_mode = st.radio(
        LANGUAGE_STRINGS[lang]["answer_mode_label"],
        LANGUAGE_STRINGS[lang]["answer_mode_options"],
        key="answer_mode"
    )
    show_reasoning = st.checkbox(
        LANGUAGE_STRINGS[lang]["show_reasoning_label"],
        key="show_reasoning"
    )
    provider = st.radio(
        LANGUAGE_STRINGS[lang]["provider_label"],
        LANGUAGE_STRINGS[lang]["provider_options"],
        key="provider"
    )
    analyze_btn = st.button(LANGUAGE_STRINGS[lang]["analyze_button"])
    clear_btn = st.button(LANGUAGE_STRINGS[lang]["clear_button"], on_click=clear_chat_history)



# Preview file upload
if uploaded_files:
    st.subheader(LANGUAGE_STRINGS[lang]["preview_header"])
    for f in uploaded_files:
        b = f.getvalue()  
        text, preview_info = get_text_from_file(b, f.name)
        show_preview_from_file(b, f.name, preview_info)

# Khi bấm nút Phân tích
if analyze_btn:
    if not uploaded_files:
        st.error(LANGUAGE_STRINGS[lang]["error_no_file"])
    else:
        with st.spinner("Đang phân tích..." if lang == "vi" else "Analyzing..."):
            all_text = ""
            for f in uploaded_files:
                b = f.getvalue()
                text, preview_info = get_text_from_file(b, f.name)
                all_text += text + "\n\n"
            if all_text.strip():
                chunks = get_text_chunks(all_text, chunk_size, chunk_overlap)
                if chunks:
                    get_vector_store(chunks, session_dir, provider)
                else:
                    st.error(LANGUAGE_STRINGS[lang]["error_no_chunks"])
            else:
                st.error(LANGUAGE_STRINGS[lang]["error_no_content"])

# QA area
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Hiển thị toàn bộ lịch sử hội thoại
for chat in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# Ô nhập câu hỏi kiểu ChatGPT
if prompt := st.chat_input(LANGUAGE_STRINGS[lang]["chat_input_placeholder"]):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi xử lý trả lời
    answer = user_input(prompt, session_dir, answer_mode, show_reasoning, provider)
    with st.chat_message("assistant"):
        st.markdown(answer)