import pandas as pd
from shared import app_dir, df
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지



df.head()
df.columns
df.shape
info_df = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
    "Null Count": [df[col].isnull().sum() for col in df.columns],
    "Dtype": [df[col].dtype for col in df.columns]
})
# 데이터 info
info_df


# 결측치 존재하는 칼럼 info만 찍기
missing_df = info_df[info_df["Null Count"] > 0].sort_values(by="Null Count", ascending=False)
missing_df

df.columns

# id line name mold_name 칼럼 삭제
df = df.drop(columns=['id','line','name','mold_name'])
df = df.drop(columns=['emergency_stop'])

# molten_temp 보간법
def fill_molten_temp_grouped(df):
    filled = []
    for time_val, group in df.groupby('time'):
        group = group.copy()
        first_index = group.index[0]
        for idx in group.index[1:]:
            if pd.isna(df.loc[idx, 'molten_temp']):
                df.loc[idx, 'molten_temp'] = df.loc[idx - 1, 'molten_temp']
        filled.append(group)
    return df

# 적용
df = fill_molten_temp_grouped(df)
# 위 보간법 적용 이후 결측치 1개에 대해 행 삭제
df = df.dropna(subset=['molten_temp'])

# production_cycletime 값 0인 행 확인후 삭제
df[df['production_cycletime'] == 0]
df = df[df['production_cycletime'] != 0]

# low_section_speed 값 65535인 행 확인 후 삭제
df[df['low_section_speed'] == 65535]
df = df[df['low_section_speed'] != 65535]


# molten_volume 결측치 가장 최근 값으로 대체
df['molten_volume'].isna().sum()
df['molten_volume'] = df['molten_volume'].fillna(method='ffill')


# cast_pressure 값이 200이하인데 양품인 25개 행 이상치라고 판단하여 행 삭제
df[(df['passorfail'] == 0) & (df['cast_pressure'] <= 200)]
df = df[~((df['passorfail'] == 0) & (df['cast_pressure'] <= 200))]


# upper_mold_temp1 값이 1449인 행 1개 확인 후 삭제
df[df['upper_mold_temp1'] == 1449]
df = df[df['upper_mold_temp1'] != 1449]


# upper_mold_temp2 값이 4232인 행 1개 확인 후 삭제
df[df['upper_mold_temp2'] == 4232]
df = df[df['upper_mold_temp2'] != 4232]


# upper_mold_temp3, lower_mold_temp3 칼럼 제거
df = df.drop(columns=['upper_mold_temp3','lower_mold_temp3'])

# sleeve_temperature 값 1449인 행 57개 확인 후 삭제
df[df['sleeve_temperature'] == 1449]
df = df[df['sleeve_temperature'] != 1449]

# physical_strength 값 65535인 행 확인 후후 제거
df[df['physical_strength']==65535]
df = df[df['physical_strength'] != 65535]


# Coolant_temperature 값 1449인 행 9개 확인 후 제거
df[df['Coolant_temperature']== 1449]
df = df[df['Coolant_temperature'] != 1449]


# EMS_operation_time 값 0인 행 1개 확인 후 제거
# 근데 위 전처리 과정에서 이미 해당 행 사라진듯?
df[df['EMS_operation_time']== 0]
df = df[df['EMS_operation_time'] != 0]

# heating_furnace 결측치 unkown으로 대체
df['heating_furnace'].isna().sum()
df['heating_furnace'] = df['heating_furnace'].fillna('Unknown')


# 결측치가 1개인 행 제거 (예: index 19327)
# 근데 위 전처리 과정에서 이미 해당 행 사라진듯?
row_to_drop = df[df[['working','low_section_speed', 'high_section_speed',
                     'cast_pressure','biscuit_thickness','upper_mold_temp1','upper_mold_temp2',
                     'lower_mold_temp1','lower_mold_temp2','sleeve_temperature',
                     'physical_strength','Coolant_temperature',
                     'molten_temp']].isnull().any(axis=1)].index
df = df.drop(index=row_to_drop)

len(df)






# tryshot_signal 결측치 unkown으로 대체
df['tryshot_signal'].isna().sum()
df['tryshot_signal'] = df['tryshot_signal'].fillna('Unknown')

df.to_csv("df_final.csv")