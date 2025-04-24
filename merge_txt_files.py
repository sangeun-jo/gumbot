import os

# data 폴더 경로
data_folder = 'data'
# 결과 파일 경로
output_file = 'merged_data.txt'

# 결과를 저장할 리스트
merged_content = []

# data 폴더의 모든 txt 파일을 읽습니다
for filename in sorted(os.listdir(data_folder), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_folder, filename)
        # 파일명에서 확장자를 제거합니다
        file_title = os.path.splitext(filename)[0]
        
        # 파일 내용을 읽습니다
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        # 파일 제목과 내용을 결합합니다
        formatted_content = f"{file_title}. {file_content}\n\n"
        merged_content.append(formatted_content)

# 결합된 내용을 새 파일에 저장합니다
with open(output_file, 'w', encoding='utf-8') as output:
    output.writelines(merged_content)

print(f"모든 txt 파일이 {output_file}에 성공적으로 합쳐졌습니다.") 