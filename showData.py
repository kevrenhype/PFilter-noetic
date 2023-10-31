# 打开文件并以二进制模式读取
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
def format_func(value, tick_number):
    return f'{value:.2f}'
with open('PFilter-noetic/observe1.dat', 'rb') as file:
# with open('PFilter-noetic/pointdis1.dat', 'rb') as file:
    # 读取文件内容并解码
    file_content = file.read().decode('utf-8')

# 使用逗号分隔符将字符串拆分为列表
data_list = file_content.split(',')
numeric_data_list = [float(item) for item in data_list if item.strip()]
#print(numeric_data_list)
plt.figure(figsize=(15,5))
nums,bins,patches = plt.hist(numeric_data_list,bins=20,edgecolor='k')
plt.xticks(bins,bins)
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
for num,bin in zip(nums,bins):
    plt.annotate(num,xy=(bin,num),xytext=(bin+1.5,num+0.5))
plt.show()
# 打印解析后的数据

