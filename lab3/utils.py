import pandas as pd

class data_utils:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature = df.select_dtypes(exclude=['object']).columns
        self.df = self.df.drop('id', axis=1)

    def drop_col(self, col:str):
        self.df = self.df.drop(col, axis=1)

    def date_init(self):
        self.df['issueDate'] = pd.to_datetime(self.df['issueDate'], format='%Y-%m-%d')
        min_issue = self.df['issueDate'].min()
        self.df['issueDate'] = (self.df['issueDate'] - min_issue).dt.days

        self.df['earliesCreditLine'] = pd.to_datetime(self.df['earliesCreditLine'], format='%b-%Y')
        min_date = self.df['earliesCreditLine'].min()
        self.df['earliesCreditLine'] = (self.df['earliesCreditLine'].dt.year - min_date.year) * 12 + (self.df['earliesCreditLine'].dt.month - min_date.month) 

    def grade(self):
        self.df = self.df.drop('subGrade', axis=1)
        def grade2int(x):
            if pd.isnull(x):
                return x
            else:
                return ord(x[0]) - ord('A') + 1

        self.df['grade'] = self.df['grade'].apply(grade2int)

    def employmentLength(self):
        import re 
        def emp_length(x):
            pattern = r'(.*?)(?=\s*years?)'
            if pd.isnull(x):
                return x
            if re.match(pattern, x):
                year = re.match(pattern, x).group(1)
                if year == '< 1':
                    return 1
                elif year == '10+':
                    return 12
                else:
                    return int(year)+1
        self.df['employmentLength'] = self.df['employmentLength'].apply(emp_length)
        
    def z_score(self):
        large_value_columns = [
        'loanAmnt', 
        'installment',
        'annualIncome',
        'revolBal',
        'ficoRangeLow',
        'ficoRangeHigh',
        "employmentTitle",
        "issueDate",
        "postCode",
        "earliesCreditLine"
        ]
        mean = self.df[large_value_columns].mean()
        std = self.df[large_value_columns].std()
        self.df[large_value_columns] = (self.df[large_value_columns] - mean) / std

    def pipeline(self):
        self.date_init()
        self.grade()
        self.employmentLength()
        self.z_score()
        self.df = self.df.astype(float)
        return self.df






        
        

    