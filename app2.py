import pickle

clf = pickle.load(open('clf.pkl' , 'rb'))
tfidf = pickle.load(open('tfidf.pkl' , 'rb'))
cleanResume = pickle.load(open('cleanResume.pkl') , 'rb')

myresume = """Name: John Smith
Email: john.smith@email.com
Phone: +1 555 234 7890
Location: New York, USA

Professional Summary:
Passionate Java Developer with 2+ years of experience in building scalable backend systems and enterprise applications using Java, Spring Boot, and REST APIs. Strong understanding of object-oriented design and database management.

Technical Skills:
- Programming Languages: Java, Python, SQL
- Frameworks: Spring Boot, Hibernate, Maven
- Databases: MySQL, PostgreSQL
- Tools: Git, IntelliJ IDEA, Docker
- Web Technologies: HTML, CSS, JavaScript

Experience:
Software Developer — Tech Solutions Inc.
Jan 2022 – Present
- Designed and implemented RESTful APIs for employee management system.
- Improved database query performance by 20%.
- Collaborated with frontend team for API integration.

Education:
Bachelor of Computer Science — XYZ University (2020)

Projects:
- Library Management System (Java + MySQL)
- Student Attendance Portal (Spring Boot + REST API)

Certifications:
- Oracle Certified Java Programmer (OCJP)

Hobbies:
Coding, Problem Solving, Reading Tech Blogs"""

cleaned_resume = cleanResume(myresume)

input_features = tfidf.transform([cleaned_resume])
prediction_id = clf.predict(input_features)[0]

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
print(prediction_id)