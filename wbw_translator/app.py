import streamlit as st
import requests
import time

st.set_page_config(layout="wide")

def tokenize_korean(text):
    """Tokenize Korean text by splitting on spaces"""
    # Simple space-based tokenization
    words = text.split()
    return words



def translate_with_google(word):
    try:
        if not word.strip():
            return ""

        # Add rate limiting delay
        time.sleep(0.1)

        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'ko',
            'tl': 'en',
            'dt': 't',
            'q': word
        }

        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 429:
            return "[Rate limited - please wait]"

        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                return result[0][0][0]
            else:
                return f"[No translation: {word}]"
        else:
            return f"[Error {response.status_code}: {word}]"

    except requests.exceptions.Timeout:
        return f"[Timeout: {word}]"
    except requests.exceptions.RequestException as e:
        return f"[Network Error: {word}]"
    except Exception as e:
        return f"[Error: {word}]"

def create_translation_display(original_text, translations):
    """Create HTML table display with translations aligned above words in multiple rows"""

    # Set words per row
    words_per_row = 10
    total_words = len([w for w in original_text if w.strip()])
    num_rows = (total_words + words_per_row - 1) // words_per_row

    # Create table with multiple rows - simplified structure
    html = '<table style="border-collapse: collapse; width: 100%;">'

    for row in range(num_rows):
        start_idx = row * words_per_row
        end_idx = min(start_idx + words_per_row, len(original_text))

        # Translation row
        html += '<tr>'
        for i in range(start_idx, end_idx):
            word = original_text[i]
            translation = translations[i]

            if word.strip():
                style = "background-color: #fff3e0; color: #333;"

                html += f'<td style="padding: 6px; text-align: center; {style} min-width: 80px;">'
                html += f'<div style="font-size: 12px;">{translation}</div>'
                html += '</td>'

                # Add extra empty cell after sentence-ending punctuation
                if word.strip() and word.rstrip().endswith(('.', '?', '!')):
                    html += '<td style="padding: 6px; min-width: 80px;"></td>'
            else:
                html += '<td style="padding: 6px; min-width: 80px;"></td>'
        html += '</tr>'

        # Korean text row
        html += '<tr>'
        for i in range(start_idx, end_idx):
            word = original_text[i]
            if word.strip():
                html += f'<td style="padding: 8px; text-align: center; font-weight: bold; font-size: 16px; border-top: 1px solid #ddd;">{word}</td>'

                # Add extra empty cell after sentence-ending punctuation
                if word.strip() and word.rstrip().endswith(('.', '?', '!')):
                    html += '<td style="padding: 8px; border-top: 1px solid #ddd;"></td>'
            else:
                html += '<td style="padding: 8px; border-top: 1px solid #ddd;"></td>'
        html += '</tr>'

        html += '<tr><td colspan="100" style="padding: 15px; background-color: #f9f9f9;"></td></tr>'

    html += '</table></div>'
    return html

def main():
    st.title("Korean Word By Word Translator")
    st.write("Translates Korean text word by word using Google Translate")

    # Text input area
    korean_text = st.text_area(
        "Korean Text:",
        height=100,
        placeholder="안녕하세요! 이 앱은 한국어에서 영어로 한 단어씩 번역됩니다"
    )

    translate_clicked = st.button("Translate", type="primary")

    if translate_clicked:
        if korean_text.strip():
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            live_results = st.empty()

            # Step 1: Tokenization
            status_text.text("Tokenizing Korean text...")
            progress_bar.progress(0.1)

            words = tokenize_korean(korean_text)
            total_words = len([w for w in words if w.strip()])

            if total_words == 0:
                st.warning("No words found to translate.")
                return

            progress_bar.progress(0.2)
            status_text.text(f"Found {total_words} words to translate. Starting translation...")

            # Step 2: Translate words one by one
            translations = []
            live_output = ""

            for i, word in enumerate(words):
                if word.strip():
                    # Update progress
                    progress = 0.2 + (i / total_words) * 0.8
                    progress_bar.progress(progress)

                    # Update status
                    status_text.text(f"Translating word {i+1} of {total_words}: {word}")

                    # Translate the word using Google Translate
                    translation = translate_with_google(word)
                    translations.append(translation)

                    # Add to live output
                    live_output += f"**{word}** → {translation}\n\n"
                    live_results.markdown(live_output)
                else:
                    translations.append("")

            # Step 3: Final display
            progress_bar.progress(1.0)
            status_text.text("Translation complete!")

            # Display final formatted result
            st.subheader("Translation Result:", anchor="translation-result")
            html_output = create_translation_display(words, translations)
            st.markdown(html_output, unsafe_allow_html=True)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        else:
            st.warning("Please enter some Korean text to translate.")

if __name__ == "__main__":
    main()
