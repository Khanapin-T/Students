import pandas as pd
df = pd.read_csv('train.csv')

df.drop(['id', 'bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 
         'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name',
         'career_start', 'career_end'], axis = 1, inplace = True)

'''Меньшинство парней получили высшее образование на иностранном языке'''

def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)


df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)


def ed_st_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    if edu_status == "Student (Specialist)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1
    if edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    if edu_status == 'PhD' or edu_status == 'Candidate of Sciences':
        return 3
df['education_status'] = df['education_status'].apply(ed_st_apply)


def ln_apply(langs):
    if langs.find('Русский') != -1:
        return 1
    return 0
df['langs'] = df['langs'].apply(ln_apply)


def ocu_tip_apply(ocu_type):
    if ocu_type == 'university':
        return 1
    return 0
df['occupation_type'] = df['occupation_type'].apply(ocu_tip_apply)

fem_phd = 0
male_phd = 0
lang_rus = 0
lang_other = 0

def edu_3lvl_sex(row):
    global fem_phd, male_phd, lang_rus, lang_other
    if row['education_status'] == 3:
        if row['sex'] == 0:
            if row['langs'] == 1:
                lang_rus += 1
            else:
                lang_other += 1
            male_phd += 1
        else:
            fem_phd += 1
    return False
df['sex'] == df.apply(edu_3lvl_sex, axis = 1)

# print('Кол-во парней с высшим образованием:', male_phd)
# print('Знают русский язык:', lang_rus)
# print('Остальные языки:', lang_other)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))