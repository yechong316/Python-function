def replace_company(strings, labels, company):

    # 组织机构label定义为 B-organization, I-organization
    b_company_label = 'B-organization'
    i_company_label = 'I-organization'
    import numpy as np
    length = len(company)
    index = np.random.randint(0, length)
    target_company = company[index]
    if len(target_company) == 1:
        insert_labels = ['B-organization']
    else:
        insert_labels = ['B-organization']
        insert_labels.extend(['I-organization'] * (len(target_company) - 1))

    # 当company是空，或者labels里面没有组织机构名的label，输入的字符串为空，那么直接返回字符串
    if company is None or b_company_label not in labels or i_company_label not in labels or strings is None: return strings, labels

    else:
        new_strings = []
        new_labels = []
        # for i in range():
        i = 0
        while i < len(labels) - 1:

            # 找到组织结构名
            if labels[i] == 'B-organization':

                # 记录其实标记索引
                while labels[i + 1] == 'I-organization':

                    i += 1
                # 记录终止索引

                new_strings.append(target_company)
                new_labels.append(insert_labels)
            else:
                new_strings.append(strings[i])
                new_labels.append(labels[i])
            i += 1

        return new_strings, new_labels