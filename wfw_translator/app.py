import streamlit as st
import requests
import time

st.set_page_config(page_title="Word-For-Word Translator", page_icon="🗺️", layout="wide")

# Configuration constants
WORDS_PER_ROW = 10
RATE_LIMIT_DELAY = 0.1
TRANSLATION_TIMEOUT = 10
MIN_CELL_WIDTH = "80px"
PROGRESS_UPDATE_INTERVAL = 0.1

# Table styling constants
TABLE_STYLE = "border-collapse: collapse; width: 100%; margin: 0; padding: 0;"
TRANSLATION_CELL_STYLE = "padding: 0; margin: 0; text-align: center; background-color: #fff3e0; color: #333; min-width: 80px; white-space: nowrap;"
ORIGINAL_CELL_STYLE = "padding: 0; margin: 0; text-align: center; font-weight: bold; font-size: 16px; border-top: 1px solid #ddd; white-space: nowrap;"
EMPTY_CELL_STYLE = "padding: 0; margin: 0; min-width: 80px;"
SPACER_ROW_STYLE = "padding: 0; margin: 0; background-color: #f9f9f9; height: 10px;"
TRANSLATION_TEXT_STYLE = "font-size: 12px; margin: 0; padding: 2px 4px;"

# Language code mappings for Google Translate
LANGUAGE_CODES = {
    'Arabic': 'ar',
    'Bulgarian': 'bg',
    'Chinese (Simplified)': 'zh-CN',
    'Chinese (Traditional)': 'zh-TW',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Estonian': 'et',
    'Finnish': 'fi',
    'French': 'fr',
    'German': 'de',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Malay': 'ms',
    'Norwegian': 'no',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Spanish': 'es',
    'Swedish': 'sv',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Vietnamese': 'vi'
}

def tokenize_text(text):
    """Tokenize text by splitting on spaces"""
    if not text or not text.strip():
        return []

    # Replace line breaks with spaces to avoid empty boxes
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Simple space-based tokenization
    words = text.split()
    return words

def validate_language_selection(source_lang, target_lang):
    """Validate that source and target languages are different and valid"""
    if source_lang == target_lang:
        return False, "Source and target languages must be different"
    if source_lang not in LANGUAGE_CODES:
        return False, f"Invalid source language: {source_lang}"
    if target_lang not in LANGUAGE_CODES:
        return False, f"Invalid target language: {target_lang}"
    return True, ""

def validate_input_text(text):
    """Validate input text for translation"""
    if not text or not text.strip():
        return False, "Please enter some text to translate"

    # Check for reasonable length limits
    if len(text.strip()) > 50000:  # ~10k words max
        return False, "Text is too long. Please limit to 50,000 characters"

    return True, ""



def get_translation_cache_key(word, source_lang, target_lang):
    """Generate a cache key for translation"""
    return f"{word.strip().lower()}|{source_lang}|{target_lang}"

def translate_with_google(word, source_lang='ko', target_lang='en'):
    try:
        if not word.strip():
            return ""

        # Check cache first
        cache_key = get_translation_cache_key(word, source_lang, target_lang)
        if cache_key in st.session_state.translation_cache:
            return st.session_state.translation_cache[cache_key]

        # Add rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)

        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': source_lang,
            'tl': target_lang,
            'dt': 't',
            'q': word
        }

        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, params=params, headers=headers, timeout=TRANSLATION_TIMEOUT)

        if response.status_code == 429:
            return "[Rate limited - please wait]"

        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                translation = result[0][0][0]
                # Cache the successful translation
                st.session_state.translation_cache[cache_key] = translation
                return translation
            else:
                error_msg = f"[No translation: {word}]"
                st.session_state.translation_cache[cache_key] = error_msg
                return error_msg
        else:
            error_msg = f"[Error {response.status_code}: {word}]"
            st.session_state.translation_cache[cache_key] = error_msg
            return error_msg

    except requests.exceptions.Timeout:
        error_msg = f"[Timeout: {word}]"
        st.session_state.translation_cache[cache_key] = error_msg
        return error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"[Network Error: {word}]"
        st.session_state.translation_cache[cache_key] = error_msg
        return error_msg
    except Exception as e:
        error_msg = f"[Error: {word}]"
        st.session_state.translation_cache[cache_key] = error_msg
        return error_msg

def calculate_table_structure(words):
    """Pre-calculate the complete table structure accounting for punctuation cells"""
    if not words:
        return []

    rows = []
    current_row = []
    current_row_cells = 0

    for word in words:
        if word.strip():
            current_row.append(('word', word))
            current_row_cells += 1

            # Add extra empty cell after sentence-ending punctuation
            if word.strip() and word.rstrip().endswith(('.', '?', '!')):
                current_row.append(('punctuation_space', ''))
                current_row_cells += 1
        else:
            current_row.append(('empty', ''))
            current_row_cells += 1

        # If we've reached the row limit, start a new row
        if current_row_cells >= WORDS_PER_ROW:
            rows.append(current_row)
            current_row = []
            current_row_cells = 0

    # Add the last row if it has content
    if current_row:
        rows.append(current_row)

    return rows

def create_translation_cell(word, translation):
    """Create a single translation cell with proper styling"""
    return f'<td style="{TRANSLATION_CELL_STYLE}"><div style="{TRANSLATION_TEXT_STYLE}">{translation}</div></td>'

def create_original_cell(word):
    """Create a single original text cell with proper styling"""
    return f'<td style="{ORIGINAL_CELL_STYLE}">{word}</td>'

def create_empty_cell():
    """Create an empty cell with proper styling"""
    return f'<td style="{EMPTY_CELL_STYLE}"></td>'

def create_spacer_row():
    """Create a spacer row for table formatting"""
    return f'<tr><td colspan="100" style="{SPACER_ROW_STYLE}"></td></tr>'

def create_partial_table(words, translations):
    """Create partial HTML table for live building during translation"""
    if not words:
        return ""

    table_structure = calculate_table_structure(words)
    html_parts = [f'<table style="{TABLE_STYLE}">']

    for row in table_structure:
        # Translation row
        html_parts.append('<tr>')
        for cell_type, word in row:
            if cell_type == 'word':
                # Find the translation for this word
                translation = ""
                try:
                    word_index = words.index(word)
                    if word_index < len(translations):
                        translation = translations[word_index]
                except ValueError:
                    pass  # Word not found in words list

                html_parts.append(create_translation_cell(word, translation))
            else:  # punctuation_space or empty
                html_parts.append(create_empty_cell())
        html_parts.append('</tr>')

        # Original text row
        html_parts.append('<tr>')
        for cell_type, word in row:
            if cell_type == 'word':
                html_parts.append(create_original_cell(word))
            else:  # punctuation_space or empty
                html_parts.append(create_empty_cell())
        html_parts.append('</tr>')

        # Spacer row
        html_parts.append(create_spacer_row())

    html_parts.append('</table>')
    return ''.join(html_parts)

def main():
    st.title("Word-For-Word Translator")
    st.write("Translates text Word-For-Word using Google Translate")

    # Language selection
    col1, col2 = st.columns(2)

    with col1:
        source_language = st.selectbox(
            "Source Language:",
            options=list(LANGUAGE_CODES.keys()),
            index=list(LANGUAGE_CODES.keys()).index('Korean')  # Default to Korean
        )

    with col2:
        target_language = st.selectbox(
            "Target Language:",
            options=list(LANGUAGE_CODES.keys()),
            index=list(LANGUAGE_CODES.keys()).index('English')  # Default to English
        )

    # Text input area
    input_label = f"{source_language} Text:"
    placeholder_text = {
        'Korean': "안녕하세요! 이 앱은 한국어에서 영어로 한 단어씩 번역됩니다",
        'Spanish': "¡Hola! Esta aplicación traduce texto palabra por palabra",
        'French': "Bonjour! Cette application traduit le texte mot par mot",
        'German': "Hallo! Diese App übersetzt Text Wort für Wort",
        'Japanese': "こんにちは！このアプリはテキストを単語ごとに翻訳します",
        'Chinese (Simplified)': "你好！这个应用程序逐字翻译文本",
        'Russian': "Привет! Это приложение переводит текст слово за словом"
    }.get(source_language, f"Enter {source_language} text to translate...")

    input_text = st.text_area(
        input_label,
        height=100,
        placeholder=placeholder_text
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        translate_clicked = st.button("Translate", type="primary")

    with col2:
        # Enable stop button only when translation is running
        stop_disabled = not st.session_state.get('translation_running', False)
        stop_clicked = st.button("Stop", type="secondary", disabled=stop_disabled)

    # Initialize session state
    if 'stop_translation' not in st.session_state:
        st.session_state.stop_translation = False
    if 'translation_running' not in st.session_state:
        st.session_state.translation_running = False
    if 'translation_cache' not in st.session_state:
        st.session_state.translation_cache = {}
    if 'partial_words' not in st.session_state:
        st.session_state.partial_words = []
    if 'partial_translations' not in st.session_state:
        st.session_state.partial_translations = []

    # Handle stop button click - this needs to be checked before translation starts
    if stop_clicked and st.session_state.get('translation_running', False):
        st.session_state.stop_translation = True
        st.warning("Translation stopped by user.")

    if translate_clicked:
        # Validate inputs before starting translation
        is_valid_lang, lang_error = validate_language_selection(source_language, target_language)
        if not is_valid_lang:
            st.error(lang_error)
            return

        # If input is empty, use placeholder text for translation
        text_to_translate = input_text.strip() if input_text.strip() else placeholder_text

        is_valid_text, text_error = validate_input_text(text_to_translate)
        if not is_valid_text:
            st.error(text_error)
            return

        # Set translation as running and trigger rerun to update button state
        st.session_state.translation_running = True
        st.session_state.stop_translation = False
        st.rerun()

    # Handle the actual translation process
    if st.session_state.get('translation_running', False) and not st.session_state.get('stop_translation', False):
        # Get language codes
        source_lang_code = LANGUAGE_CODES[source_language]
        target_lang_code = LANGUAGE_CODES[target_language]

        # If input was empty, use placeholder text for translation
        text_to_translate = input_text.strip() if input_text.strip() else placeholder_text

        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_results = st.empty()
        final_results = st.empty()

        # Step 1: Tokenization
        status_text.text(f"Tokenizing {source_language} text...")
        progress_bar.progress(0.1)

        words = tokenize_text(text_to_translate)
        total_words = len([w for w in words if w.strip()])

        if total_words == 0:
            st.warning("No words found to translate.")
            st.session_state.translation_running = False
            st.rerun()
            return

        progress_bar.progress(0.2)
        status_text.text(f"Found {total_words} words to translate. Starting translation...")

        # Step 2: Translate words one by one with live table building
        translations = []
        translation_stopped = False

        for i, word in enumerate(words):
            # Check if stop was requested
            if st.session_state.get('stop_translation', False):
                translation_stopped = True
                break

            if word.strip():
                # Update progress
                progress = 0.2 + (i / total_words) * 0.8
                progress_bar.progress(progress)

                # Update status
                status_text.text(f"Translating word {i+1} of {total_words}: {word}")

                # Translate the word using Google Translate
                translation = translate_with_google(word, source_lang_code, target_lang_code)
                translations.append(translation)

                # Store partial results in session state for preservation
                st.session_state.partial_words = words[:i+1]
                st.session_state.partial_translations = translations.copy()

                # Build and display partial table live
                partial_table_html = create_partial_table(words[:i+1], translations)
                live_results.markdown(partial_table_html, unsafe_allow_html=True)
            else:
                translations.append("")

        # Step 3: Final display
        if not translation_stopped:
            progress_bar.progress(1.0)
            status_text.text("Translation complete!")
        else:
            progress_bar.progress(0.2 + (len(translations) / total_words) * 0.8)
            status_text.text(f"Translation stopped at word {len(translations)} of {total_words}")

        # Clear progress indicators and mark translation as complete
        progress_bar.empty()
        status_text.empty()
        st.session_state.translation_running = False

        # Display final results without clearing them
        final_table_html = create_partial_table(words, translations)
        final_results.markdown("### Translation Results")
        final_results.markdown(final_table_html, unsafe_allow_html=True)

        if not translation_stopped:
            final_results.success(f"Successfully translated {total_words} words from {source_language} to {target_language}")
        else:
            final_results.warning(f"Translation stopped after {len(translations)} words")
    elif st.session_state.get('stop_translation', False):
        # Create a permanent display area for partial results
        st.markdown("### Partial Translation Results")
        if 'partial_translations' in st.session_state and 'partial_words' in st.session_state:
            partial_table_html = create_partial_table(st.session_state.partial_words, st.session_state.partial_translations)
            st.markdown(partial_table_html, unsafe_allow_html=True)
            st.success(f"Translation stopped after {len(st.session_state.partial_translations)} words")

        # st.warning("Translation stopped by user.")
        st.session_state.translation_running = False
        st.session_state.stop_translation = False
        # Don't call st.rerun() here to preserve the displayed results

if __name__ == "__main__":
    main()
