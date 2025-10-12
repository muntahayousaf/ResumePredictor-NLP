import pickle
import nltk
import streamlit as st
import re

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl' , 'rb'))
tfidf = pickle.load(open('tfidf.pkl' , 'rb'))

def cleanresume(txt):
    cleanText = re.sub(r'http\S+|www\S+', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\d+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    cleanText = cleanText.lower()
    return cleanText


def main():
    st.title("Resume Category Prediction App")
    uploaded_file = st.file_uploader('Upload Resume' , type=['txt' , 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanresume([resume_text])
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

        category_mapping = {
        15: "Java Developer" , 
        23: "Testing" ,
        8: "DevOps Engineer" ,
        20: "Python Developer" ,
        24: "Web Designing" ,
        12: "HR" ,
        13: "Hadoop" ,
        22: "Sales" ,
        6: "Data Science" ,
        16: "Mechanical Engineer" ,
        10: "ETL Developer" ,
        3: "Blockchain" ,
        18: "Operations Manager" ,
        1: "Arts" ,
        7: "Database" ,
        14: "Health and fitness" ,
        19: "PMO" ,
        11: "Electrical Engineeringr" ,
        4: "Business Analystr" ,
        9: "DotNet Developer" ,
        2: "Automation Testing" ,
        17: "Network Security Engineer" ,
        5: "Civil Engineer" ,
        21: "SAP Developer" ,
        0: "Advocate" ,
        }

        category_name = category_mapping.get(prediction_id , 'Unknown')

        st.write("Predicted Category:" , category_name)

# python main
if __name__ == "__main__":
    main()