#### Introduction 
In this data-driven age, the protection of Personally Identifiable Information (PII) has become critically important, especially when it’s 
embedded in the extensive personal details often found in essays.  This project involves in building an end-to-end strategy to detect sensitive personal identifiers from student essays.

## Requirements
<>

## Data Pipeline
Data is featured by The Learning Agency Lab in Kaggle.
> The goal of this competition is to develop a model that detects personally identifiable information (PII) in student writing. Your efforts to automate the detection and removal of PII from educational data will lower the cost of releasing educational datasets. This will support learning science research and the development of educational tools. Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.
https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data

### Data Card
| Feature | Type | Description |
|----------------|------|-----------------|
| Document | int  | An integer ID for each essay (primary key) |
| full_text | string  | UTF-8 representation of the essay  |
| tokens | list  | Each word of string type stored as a list  |
| trailing_whitespace | list | List of boolean values including whether each token is followed by a whitespace |
| labels | list | Token labels in BIO format |

### Label Classes
The labels being the target feature which has different classes to predict,
| Label Class | Description |
|---|-----------------|
| NAME_STUDENT | The full or partial name of a student who is not necessarily the author of the essay. This excludes instructors, authors, and other person names|
| EMAIL | A student’s email address|
| USERNAME | A student's username on any platform. |
| ID_NUM | A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number.|
|PHONE_NUM | A phone number associated with a student |
| URL_PERSONAL | A URL that might be used to identify a student.|
| STREET_ADDRESS | This holds the student’s address.|

## Downloading Dataset
`downloadData.py` downloads the zipped source files including test from the source
`unzipFile.py` extracts the downloaded datasets (files are stored in .json)

