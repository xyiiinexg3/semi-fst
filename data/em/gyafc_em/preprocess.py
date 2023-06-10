file_path = "D:\\1.Workstation\\VSCodeProject\\11.semiFST\\semi-fst\\data\\em\\gyafc_em\\test.tgt"

# 打开 val.tgt 文件进行读取
with open(file_path, 'r', encoding='UTF-8') as file:
    lines = file.readlines()

# 提取每一行的第n个元素并写入对应的文件
import ast

for n in range(4):
    ref_file = f'D:\\1.Workstation\\VSCodeProject\\11.semiFST\\semi-fst\\data\\em\\test\\formal.ref{n}'
    with open(ref_file, 'w',encoding="UTF-8") as f:
        for line in lines:
            elements = ast.literal_eval(line.strip())
            element = elements[n]
            f.write(str(element) + '\n')
