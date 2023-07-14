from PyPDF2 import PdfReader
import os
import glob

directory_path = r'C:\Users\aryan\Downloads\plato-gpt\datasets\sam_harris_podcast_transcripts'
pdf_files = glob.glob(os.path.join(directory_path, '*.pdf')) # get all dataset chunks

#get the total size of dataset
text = ''
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    pages = reader.pages

    # extracting text from page
    for page in pages:
        text += page.extract_text()

with open(r"C:\Users\aryan\Downloads\plato-gpt\datasets\sam_harris_podcast_transcripts/data.txt", "w") as f:
    f.write(text)

with open(r"C:\Users\aryan\Downloads\plato-gpt\datasets\sam_harris_podcast_transcripts/data.txt", "r") as f:
    text = f.read()

print(text[:20])