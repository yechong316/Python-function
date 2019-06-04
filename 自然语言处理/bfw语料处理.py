

import re
from tqdm import tqdm


def file_lines(file_path):
    # 将数据对齐
    with open(file_path, 'rb') as fp:
        b = fp.read()
    # 将语料库内容转为一行数据，用换行符链接
    content = b.decode('utf8', 'ignore')
    lines = []
    for line in tqdm(content.split('\n')):  # 将语料库内容转换按行储存列表，末尾是换行符
        try:
            line = line.replace('\n', '').strip()  # 'M 畹/华/吾/侄/'
            if line.startswith('E'):
                lines.append('')
            elif line.startswith('M '):
                chars = line[2:].split('/')  # char = ['畹', '华', '吾', '侄', '']
                while len(chars) and chars[len(chars) - 1] == '.':
                    chars.pop()
                if chars:
                    sentence = ''.join(chars)

                    # re.sub用于把sentence中的空格'' 替换成'，'
                    sentence = re.sub('\s+', '，', sentence)  # '畹华吾侄' + 表示空格
                    lines.append(sentence)
        except:
            print(line)
            return lines

        lines.append('')
    return lines

if __name__ == '__main__':

    path = r'D:\自然语言处理实战\应俊老师\Seq2Seq_Chatbot\Seq2Seq_Chatbot\db\dgk_shooter_min.conv'
    lines = file_lines(path)
    print(lines)