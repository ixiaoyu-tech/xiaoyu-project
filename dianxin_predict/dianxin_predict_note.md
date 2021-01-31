# 电信用户流失分析预测

[TOC]

### 项目背景

​	用户流失预测在机器学习中算是一种比较典型的分类场景，做好用户的流失预测可以降低营销成本，留住用户并且获得更好的用户体验，在三大巨头的瓜分下，做好营销运营比重新获取一个新用户更节省成本，同时达到较好的运营回报。

​	关于用户留存有这样一个观点，如果将用户流失率降低5%，公司利润将提升25%-85%。如今高居不下的获客成本让电信运营商遭遇“天花板”，甚至陷入获客难的窘境。如果在传统分类模式下，通常是通过人工对各个特征进行统计，然后分到合适的类别中，这样不但会耗费大量的资源，且低效。随着市场饱和度上升，电信运营商亟待解决增加用户黏性，延长用户生命周期的问题。因此，电信用户流失分析与预测至关重要。

- 数据集：[Kaggle-电信用户流失数据集](https://www.kaggle.com/blastchar/telco-customer-churn)

- 工具：Python、Tableau、Excel、sklearn等

- 按照数据分析的标准流程进行本次的数据分析:

  1. 分析用户特征与流失的关系。
  2. 从整体情况看，流失用户的普遍具有哪些特征？
  3. 尝试找到合适的模型预测流失用户。
  4. 针对性给出增加用户黏性、预防流失的建议。

- 具体实现内容包括：
  对数据进行数据预处理 包括缺失值，异常值，重复值
  描述性分析各个特征与流失用户的占比是否显著
  将连续型变量进行分箱离散化
  将离散型特征进行独热编码
  建立模型，将源数据进行标准化
  熟练运用多种分类模型对电信用户进行预测

### 1 明确问题

#### 1.1 分析目的

获取一个新客户的成本远低于挽留或者维系一个老客户的成本，如何挽留更多用户成为一项关键业务指标，为了更好运营用户，首先要了解流失用户的特征，分析流失原因，并合理预测下个阶段的预测用流失率，确定挽留目标用户群体并制定有效方案。

项目主要围绕降低电信运营商用户流失率展开，根据用户的个人情况、服务属性、合同信息展开分析，找出影响用户流失的关键因素，并建立了用户流失的分类模型，针对潜在的流失用户制定预警与召回策略。

#### 1.2 分析思路

第一部分：数据预处理

导入数据、类型转换、处理异常值

第二部分：从流失率角度进行分析

用户的个人情况、服务属性、合同信息对于流失率的影响

第三部分：从用户价值角度进行分析

用户缴费金额分布、用户累计缴费金额分布、用户终身价值(LTV)

第四部分：通过分类模型预测用户流失

特征工程、模型选择（单一模型、多模型融合）、模型评估

#### 1.3 提出问题

1. 分析用户特征与流失的关系。
2. 从整体情况看，流失用户的普遍具有哪些特征？
3. 尝试找到合适的模型预测流失用户。
4. 针对性给出增加用户黏性、预防流失的建议。

### 2 数据理解

#### 2.1 数据详情

每行代表一个客户，每列包含元数据列中描述的客户属性。一共7043行数据，21个列。前20个为特征列，最后一个为研究对象。

| 序号 |      字段名      | 数据类型 |                             解释                             |
| :--: | :--------------: | :------: | :----------------------------------------------------------: |
|  1   |    customerID    |  String  |                            用户ID                            |
|  2   |      gender      |  String  |                    客户性别（男性or女性）                    |
|  3   |  SeniorCitizen   | Integer  |                  老年人(1表示是，0表示不是)                  |
|  4   |     Partner      |  String  |                       配偶(Yes or No)                        |
|  5   |    Dependents    |  String  |                       家属(Yes or No)                        |
|  6   |      tenure      | Integer  |                    职位(0~72，共73个职位)                    |
|  7   |   PhoneService   |  String  |                    电话服务（Yes or No）                     |
|  8   |  MultipleLines   |  String  |           多线（Yes 、No or No phoneservice 三种）           |
|  9   | InternetService  |  String  |   互联网服务（No, DSL数字网络，fiber optic光纤网络 三种）    |
|  10  |  OnlineSecurity  |  String  |         在线安全（Yes，No，No internetserive 三种）          |
|  11  |   OnlineBackup   |  String  |         在线备份（Yes，No，No internetserive 三种）          |
|  12  | DeviceProtection |  String  |         设备保护（Yes，No，No internetserive 三种）          |
|  13  |   TechSupport    |  String  |         技术支持（Yes，No，No internetserive 三种）          |
|  14  |   StreamingTV    |  String  |         网络电视（Yes，No，No internetserive 三种）          |
|  15  | StreamingMovies  |  String  |         网络电影 （Yes，No，No internetserive 三种）         |
|  16  |     Contract     |  String  |       合同（Month-to-month，One year，Two year 三种）        |
|  17  | PaperlessBilling |  String  |                      账单（Yes or No）                       |
|  18  |  PaymentMethod   |  String  | 付款方式（bank transfer，credit card，electronic check，mailed check 四种） |
|  19  |  MonthlyCharges  | Integer  |                            月消费                            |
|  20  |   TotalCharges   | Integer  |                            总消费                            |
|  21  |      Churn       |  String  |                      流失（Yes or No）                       |

#### 2.2 数据预处理

数据清洗的“**完全合一**”规则:

> 1、完整性：单条数据是否存在空值，统计的字段是否完善。
> 2、 全面性：观察某一列的全部数值，通过常识来判断该列是否有问题，比如：数据定义、单位标识、数据本身。
> 3、合法性：数据的类型、内容、大小的合法性。比如数据中是否存在非ASCII字符，性别存在了未知，年龄超过了150等。
> 4、唯一性：数据是否存在重复记录，因为数据通常来自不同渠道的汇总，重复的情况是常见的。行数据、列数据都需要是唯一的。

```python
# 导入相关包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# 导入数据集
customerDF = pd.read_csv('/Users/fq/Desktop/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

查看数据集信息，查看数据集大小，并初步观察前10条的数据内容。

```python
#  查看数据集大小
customerDF.shape
# 运行结果：(7043, 21)

# 设置查看列不省略
pd.set_option('display.max_columns',None)

# 查看前5条数据
customerDF.head()
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gn0ut4ly49j31fu0csjte.jpg)

```python
# 统计缺失值
pd.isnull(customerDF).sum()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0v9426rhj30d20l8gnm.jpg" alt=" " style="zoom:50%;" />

```python
# 查看数据类型
customerDF.info()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vabpef6j30om0qqdl2.jpg" style="zoom:50%;" />

查看数据类型，根据一般经验，发现‘TotalCharges’总消费额的数据类型为字符串，应该转换为浮点型数据。将‘TotalCharges’总消费额的数据类型转换为浮点型，发现错误：字符串无法转换为数字。

```python
# customerDF[['TotalCharges']].astype(float)
# ValueError: could not convert string to float: 
```

依次检查各个字段的数据类型、字段内容和数量。最后发现“TotalCharges”（总消费额）列有11个用户数据缺失。

```python
# 依次检查各个字段的数据类型、字段内容和数量。最后发现“TotalCharges”（总消费额）列有11个用户数据缺失。
# format函数为字符串的格式化函数
for x in customerDF.columns:
    test=customerDF.loc[:,x].value_counts()
    print('{0} 的行数是：{1}'.format(x,test.sum()))
    print('{0} 的数据类型是：{1}'.format(x,customerDF[x].dtypes))
    print('{0} 的内容是：\n{1}\n'.format(x,test))
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vof5av1j30om0i6gnm.jpg" style="zoom:50%;" />

```python
# 采用强制转换，将“TotalCharges”（总消费额）转换为浮点型数据。
customerDF[['TotalCharges']]=pd.to_numeric(customerDF['TotalCharges'], errors='coerce').fillna(0)
customerDF['TotalCharges'].dtype
```

 注意在pandas中的数据类型强制转换问题，三种方法：astype（适用于无数据缺失的数据），自定义函数，pandas的函数（注意pandas的版本问题）

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vqit661j308g01gjr8.jpg" style="zoom:50%;" />

转换后发现“TotalCharges”（总消费额）列有11个用户数据缺失，为NaN。

```python
test=customerDF.loc[:,'TotalCharges'].value_counts().sort_index()
print(test.sum())
#运行结果：7032

print(customerDF.tenure[customerDF['TotalCharges'].isnull().values==True])
#运行结果：11
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vrkf9xfj30e80foaas.jpg" style="zoom:50%;" />

经过观察，发现这11个用户‘tenure’（入网时长）为0个月，推测是当月新入网用户。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vuapm0yj30ko0e6abh.jpg" style="zoom:50%;" />

```python
print(customerDF.isnull().any())
print(customerDF[customerDF['TotalCharges'].isnull().values==True][['tenure','MonthlyCharges','TotalCharges']])
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vthqhmoj30fo0pimzm.jpg" style="zoom:50%;" />

根据一般经验，用户即使在注册的当月流失，也需缴纳当月费用。因此将这11个用户入网时长改为1，将总消费额填充为月消费额，符合实际情况。

```python
#将总消费额填充为月消费额
customerDF.loc[:,'TotalCharges'].replace(to_replace=np.nan,value=customerDF.loc[:,'MonthlyCharges'],inplace=True)
#查看是否替换成功
print(customerDF[customerDF['tenure']==0][['tenure','MonthlyCharges','TotalCharges']])
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vw7kxqzj30lu0fsgn5.jpg" style="zoom:50%;" />

```python
# 将‘tenure’入网时长从0修改为1
customerDF.loc[:,'tenure'].replace(to_replace=0,value=1,inplace=True)
print(pd.isnull(customerDF['TotalCharges']).sum())
print(customerDF['TotalCharges'].dtypes)
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0vww8q8vj305g02aglf.jpg" style="zoom:50%;" />

查看数据的描述统计信息，根据一般经验，所有数据正常。

```python
# 获取数据类型的描述统计信息
customerDF.describe()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0x0b5x4aj30o00dggn3.jpg" style="zoom:50%;" />

至此，所有数据预处理完成。

### 3 数据可视化

将用户特征划分为用户属性、服务属性、合同属性，并从这三个维度进行可视化分析。

查看流失用户数量和占比

```python
# 查看流失用户数量和占比
plt.rcParams['figure.figsize']=6,6
plt.pie(customerDF['Churn'].value_counts(),labels=customerDF['Churn'].value_counts().index,autopct='%1.2f%%',explode=(0.1,0))
plt.title('Churn(Yes/No) Ratio')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0ww0qajkj30gy0i4t9x.jpg" alt="" style="zoom:50%;" />

饼状图可以内部直接算出百分比，从图中可以看出流失比例26.54%，占比较高，也处于样本不均衡问题，后面可以采用过采样来解决，过采样相比欠采样较稳定

```python
# 用户流失数据柱状图
churnDf=customerDF['Churn'].value_counts().to_frame()
x=churnDf.index
y=churnDf['Churn']
plt.bar(x,y,width = 0.5)
plt.title('Churn(Yes/No) Num')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn0x2vm7byj30ng0k6gm8.jpg" style="zoom:50%;" />

数据集分布极不均衡，流失用户占比达26.54%。

#### 3.1 用户属性分析

##### 3.1.1 按照老年人和性别比较

分别按照用户属性（男女、老少）统计用户流失数量：

```python
def barplot_percentages(feature,orient='v',axis_name="percentage of customers"):
    ratios = pd.DataFrame()
    g = (customerDF.groupby(feature)["Churn"].value_counts()/len(customerDF)).to_frame()
    g.rename(columns={"Churn":axis_name},inplace=True)
    g.reset_index(inplace=True)

    print(g)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        plt.rcParams.update({'font.size': 13})
        plt.legend(fontsize=10)
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
        plt.legend(fontsize=10)
    plt.title('Churn(Yes/No) Ratio as {0}'.format(feature))
    plt.show()
barplot_percentages("SeniorCitizen")
barplot_percentages("gender")
```

按照年龄分布分为

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn10jjjsphj30no0rgjtn.jpg" alt="" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn14bgqe72j30p20r00v3.jpg" style="zoom:50%;" />

##### 3.1.2 按性别划分老年人和年轻人

```python
# 按照老年和青年划分下，查看男女流失比例。
customerDF['churn_rate'] = customerDF['Churn'].replace("No", 0).replace("Yes", 1)
g = sns.FacetGrid(customerDF, col="SeniorCitizen", height=4, aspect=.9)
ax = g.map(sns.barplot, "gender", "churn_rate", palette = "Blues_d", order= ['Female', 'Male'])
plt.rcParams.update({'font.size': 13})
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn14ey7ms4j30wu0eydgp.jpg" style="zoom:50%;" />

##### 3.1.3 按照家属和伴侣划分

```python

fig, axis = plt.subplots(1, 2, figsize=(12,4))
axis[0].set_title("Has Partner")
axis[1].set_title("Has Dependents")
axis_y = "percentage of customers"

# Plot Partner column
gp_partner = (customerDF.groupby('Partner')["Churn"].value_counts()/len(customerDF)).to_frame()
gp_partner.rename(columns={"Churn": axis_y}, inplace=True)
gp_partner.reset_index(inplace=True)
ax1 = sns.barplot(x='Partner', y= axis_y, hue='Churn', data=gp_partner, ax=axis[0])
ax1.legend(fontsize=10)
#ax1.set_xlabel('伴侣')


# Plot Dependents column
gp_dep = (customerDF.groupby('Dependents')["Churn"].value_counts()/len(customerDF)).to_frame()
#print(gp_dep)
gp_dep.rename(columns={"Churn": axis_y} , inplace=True)
#print(gp_dep)
gp_dep.reset_index(inplace=True)
print(gp_dep)

ax2 = sns.barplot(x='Dependents', y= axis_y, hue='Churn', data=gp_dep, ax=axis[1])
#ax2.set_xlabel('家属')

# 解决中文显示问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#设置字体大小
plt.rcParams.update({'font.size': 15})
ax2.legend(fontsize=15)

#设置
plt.show()
```

![image-20210126152405062](/Users/fq/Library/Application Support/typora-user-images/image-20210126152405062.png)

##### 3.1.4 按照工作职位划分

```python
# Kernel density estimaton核密度估计
def kdeplot(feature,xlabel):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {0}".format(feature))
    ax0 = sns.kdeplot(customerDF[customerDF['Churn'] == 'No'][feature].dropna(), color= 'navy', label= 'Churn: No', shade='True')
    ax1 = sns.kdeplot(customerDF[customerDF['Churn'] == 'Yes'][feature].dropna(), color= 'orange', label= 'Churn: Yes',shade='True')
    plt.xlabel(xlabel)
    #设置字体大小
    plt.rcParams.update({'font.size': 20})
    plt.legend(fontsize=10)
kdeplot('tenure','tenure')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn14t2b9s9j311a0g0wh5.jpg" style="zoom:50%;" />

##### 3.1.5 总结

- 用户流失与性别基本无关；
- 年老用户流失占显著高于年轻用户；

- 有伴侣的用户流失占比低于无伴侣用户；
- 有家属的用户较少；有家属的用户流失占比低于无家属用户;
- 在网时长越久，流失率越低，符合一般经验；
- 在网时间达到三个月，流失率小于在网率，证明用户心理稳定期一般是三个月。

#### 3.2 服务属性分析

##### 3.2.1 按照多线特征划分

```python
plt.figure(figsize=(9, 4.5))
barplot_percentages("MultipleLines", orient="h")
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn156f3tl2j311k0ou76d.jpg" alt=" " style="zoom:50%;" />

##### 3.2.2 按照网络服务划分

```python
plt.figure(figsize=(9, 4.5))
barplot_percentages("InternetService", orient="h")
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn152ncrc7j313m0oo0vt.jpg" style="zoom:50%;" />

##### 3.2.3 互联网附加服务用户数量

```python
cols = ["PhoneService","MultipleLines","OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df1 = pd.melt(customerDF[customerDF["InternetService"] != "No"][cols])
df1.rename(columns={'value': 'Has service'},inplace=True)
plt.figure(figsize=(20, 8))
ax = sns.countplot(data=df1, x='variable', hue='Has service')
ax.set(xlabel='Internet Additional service', ylabel='Num of customers')
plt.rcParams.update({'font.size':20})
plt.legend( labels = ['No Service', 'Has Service'],fontsize=15)
plt.title('Num of Customers as Internet Additional Service')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15a7e10zj31j40mytbi.jpg" style="zoom:50%;" />

```python
plt.figure(figsize=(20, 8))
df1 = customerDF[(customerDF.InternetService != "No") & (customerDF.Churn == "Yes")]
df1 = pd.melt(df1[cols])
df1.rename(columns={'value': 'Has service'}, inplace=True)
ax = sns.countplot(data=df1, x='variable', hue='Has service', hue_order=['No', 'Yes'])
ax.set(xlabel='Internet Additional service', ylabel='Churn Num')
plt.rcParams.update({'font.size':20})
plt.legend( labels = ['No Service', 'Has Service'],fontsize=15)
plt.title('Num of Churn Customers as Internet Additional Service')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15djcs32j31j40n8q5u.jpg" style="zoom:50%;" />

##### 3.2.4 总结

- 电话服务整体对用户流失影响较小。
- 单光纤用户的流失占比较高；
- 光纤用户绑定了安全、备份、保护、技术支持服务的流失率较低；
- 光纤用户附加流媒体电视、电影服务的流失率占比较高。

#### 3.3 合同属性分析

##### 3.3.1 按照支付方式

```python
plt.figure(figsize=(9, 4.5))
barplot_percentages("PaymentMethod",orient='h')
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15gbsx86j315m0qwtdf.jpg" style="zoom:50%;" />

##### 3.3.2 按照用户账单

```python
g = sns.FacetGrid(customerDF, col="PaperlessBilling", height=6, aspect=.9)
ax = g.map(sns.barplot, "Contract", "churn_rate", palette = "Blues_d", order= ['Month-to-month', 'One year', 'Two year'])
plt.rcParams.update({'font.size':18})
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15i9pgjvj31ai0me762.jpg" style="zoom:50%;" />

##### 3.3.3 按照月消费和总消费

```python
kdeplot('MonthlyCharges','MonthlyCharges')
kdeplot('TotalCharges','TotalCharges')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15njel3oj30yk0fujus.jpg" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn15rt1fi0j30vw0getbj.jpg" style="zoom:50%;" />

##### 3.3.4 总结

- 采用电子支票支付的用户流失率最高，推测该方式的使用体验较为一般；
- 签订合同方式对客户流失率影响为：按月签订 > 按一年签订 > 按两年签订，证明长期合同最能保留客户；
- 月消费额大约在70-110之间用户流失率较高；
- 长期来看，用户总消费越高，流失率越低，符合一般经验。

#### 3.4 数据分箱离散化

##### 3.4.1 定义消费等级

```python
def charge_to_level(charge):
    if charge<=da.loc["25%"]:
        return "低消费"
    elif charge<=da.loc["50%"] and charge>da.loc["25%"]:
        return "中低消费"
    elif charge<=da.loc["75%"] and charge>da.loc["50%"]:
        return "中高消费"
    else:
        return "高消费"
da=customerDF["MonthlyCharges"].describe()
customerDF["level_MonthlyCharges"] = customerDF["MonthlyCharges"].apply(charge_to_level)
da=customerDF["TotalCharges"].describe()
customerDF["level_TotalCharges"] = customerDF["TotalCharges"].apply(charge_to_level)
display(data["level_MonthlyCharges"].value_counts())
customerDF.sample(5)
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gn5n443397j31kg0iwdj7.jpg)

##### 3.4.2 消费等级用户分布

```python
fig=plt.figure(figsize=(20,10),dpi=80)
# 子图1
fig.add_subplot(2,2,1)
sns.countplot(x="level_MonthlyCharges",hue="Churn",data=customerDF,order=["低消费","中低消费","中高消费","高消费"])

# 子图2
fig.add_subplot(2,2,2)
sns.countplot(x="level_TotalCharges",hue="Churn",data=customerDF,order=["低消费","中低消费","中高消费","高消费"])
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn5n5rdeflj31ju0dwjtr.jpg" style="zoom:50%;" />

- 对于月消费来说，流失用户主要集中在中高消费以及高消费；

- 对于总消费来说，流失用户主要集中在低消费和中低消费中。

  那么我们可以对这部分用户进行精细化运营以最大程度留住用户。

### 4 用户流失预测

对数据集进一步清洗和提取特征，通过特征选取对数据进行降维，采用机器学习模型应用于测试数据集，然后对构建的分类模型准确性进行分析。

#### 4.1 数据清洗

通过观察数据类型，发现除了“**tenure”、“MonthlyCharges”、“TotalCharges**”是连续特征，其它都是离散特征。对于连续特征，采用标准化方式处理。对于离散特征，特征之间没有大小关系，采用**one-hot编码**；特征之间有大小关联，则采用数值映射。

##### 4.1.1 删除不必要特征

```python
customerID=customerDF['customerID']
customerDF.drop(['customerID'],axis=1, inplace=True)
```

##### 4.1.2 获取离散特征

```python
cateCols = [c for c in customerDF.columns if customerDF[c].dtype == 'object' or c == 'SeniorCitizen']
dfCate = customerDF[cateCols].copy()
dfCate.head(3)
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gn5sin9yzfj31jo072wfh.jpg)

##### 4.1.3 进行特征编码

```python
for col in cateCols:
    if dfCate[col].nunique() == 2:
        dfCate[col] = pd.factorize(dfCate[col])[0]
    else:
        dfCate = pd.get_dummies(dfCate, columns=[col])
dfCate['tenure']=customerDF[['tenure']]
dfCate['MonthlyCharges']=customerDF[['MonthlyCharges']]
dfCate['TotalCharges']=customerDF[['TotalCharges']]
```

##### 4.1.4 查看关联关系

```python
plt.figure(figsize=(16,8))
dfCate.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.show()
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gn5skvjwv4j30qh0npjtu.jpg)

#### 4.2 特征选取

将最后一列'Churn'作为分类标识，丢掉不重要的特征'gender','PhoneService','OnlineSecurity','OnlineBackup'，'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies'，将其余特征作为预测特征。

##### 4.2.1 选区预测特征

```python
# 特征选择
dropFea = ['gender','PhoneService',
           'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection','TechSupport',
           'StreamingTV', 'StreamingMovies',
           ]
dfCate.drop(dropFea, inplace=True, axis =1) 
# 最后一列是作为标识
target = dfCate['Churn'].values
#列表：特征和1个标识
columns = dfCate.columns.tolist()
```

##### 4.2.2 构造训练集和测试集

调用sklearn的train_test_split进行数据集划分，将原始数据集划分称为70%的训练集和30%的测试集。

```python
# 列表：特征
columns.remove('Churn')
# 含有特征的DataFrame
features = dfCate[columns].values
# 30% 作为测试集，其余作为训练集
# random_state = 1表示重复试验随机得到的数据集始终不变
# stratify = target 表示按标识的类别，作为训练数据集、测试数据集内部的分配比例
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify = target, random_state = 1)
```

#### 4.3 建立模型

分别从sklearn导入SVC、DecisionTreeClassifier、 RandomForestClassifier、KNeighborsClassifier、AdaBoostClassifier构造支持向量机、决策树、随机森林、KNN算法分类器以及Adaboost分类器。

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier #KNN分类模型
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # 评估指标 --正确率 精准率 召回率 F1调和平均值

# 构造各种分类器
classifiers = [
    SVC(random_state = 1, kernel = 'rbf'),    
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
    RandomForestClassifier(random_state = 1, criterion = 'gini'),
    KNeighborsClassifier(metric = 'minkowski'),
    AdaBoostClassifier(random_state = 1),   
]
# 分类器名称
classifier_names = [
            'svc', 
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
            'adaboostclassifier',
]
# 分类器参数
#注意分类器的参数，字典键的格式，GridSearchCV对调优的参数格式是"分类器名"+"__"+"参数名"
classifier_param_grid = [
            {'svc__C':[0.1], 'svc__gamma':[0.01]},
            {'decisiontreeclassifier__max_depth':[6,9,11]},
            {'randomforestclassifier__n_estimators':range(1,11)} ,
            {'kneighborsclassifier__n_neighbors':[4,6,8]},
            {'adaboostclassifier__n_estimators':[70,80,90]}
]
```

#### 4.4 模型参数调优与评估

##### 4.4.1 模型参数调优

利用交叉验证对算法参数进行优化，寻找最优的参数 和最优的准确率分数，并利用predict函数进行预测，在预测过程中使用的参数均为上一步得到的最优参数。

```python
# 对具体的分类器进行 GridSearchCV 参数调优
from sklearn.model_selection import GridSearchCV #网格交叉验证
from sklearn.pipeline import Pipeline #引入流水线
from sklearn.preprocessing import StandardScaler, MinMaxScaler # StandardScaler：均值标准差标准化 # MinMaxScaler：最小最大值标准化
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 对具体的分类器进行 GridSearchCV 参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy_score'):
    response = {}
    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv=3, scoring = score)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch 最优参数：", search.best_params_)
    print("GridSearch 最优分数： %0.4lf" %search.best_score_)
    #采用predict函数（特征是测试数据集）来预测标识，预测使用的参数是上一步得到的最优参数
    predict_y = gridsearch.predict(test_x)
    print(" 准确率 %0.4lf" %accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response
 
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    #采用 StandardScaler 方法对数据规范化
    pipeline = Pipeline([
            #('scaler', StandardScaler()),
            #('pca',PCA),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')
```

##### 4.4.2 评估结果

GridSearch 最优参数： {'svcC': 0.1, 'svcgamma': 0.01}
GridSearch 最优分数： 0.7884
 准确率 0.7823**
GridSearch 最优参数： {'decisiontreeclassifiermax_depth': 6}
GridSearch 最优分数： 0.7911
 准确率 0.7695
GridSearch 最优参数： {'randomforestclassifiern_estimators': 10}
GridSearch 最优分数： 0.7840
 准确率 0.7624
GridSearch 最优参数： {'kneighborsclassifiern_neighbors': 8}
GridSearch 最优分数： 0.7917
 准确率 0.7733
**GridSearch 最优参数： {'adaboostclassifiern_estimators': 80}
GridSearch 最优分数： 0.8039
 准确率 0.7960**

根据预测结果可以得出，以上数据在使用adaboost算法时候分类的准确率最高，可以达到79.6%，而其他算法都因为自身算法的缺点，在该数据集上并没有表现出较好的分类效果。

### 5 结论与意见

根据以上分析，得到高流失率用户的特征：

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn7617t4klj30zb0u0goq.jpg" style="zoom:50%;" />

- 用户属性：老年用户，未婚用户，无亲属用户更容易流失；
- 服务属性：在网时长小于半年，有电话服务，光纤用户/光纤用户附加流媒体电视、电影服务，无互联网增值服务；
- 合同属性：签订的合同期较短，采用电子支票支付，是电子账单，月租费约70-110元的客户容易流失；
  其它属性对用户流失影响较小，以上特征保持独立。

针对上述结论，从业务角度给出相应建议：

根据预测模型，构建一个高流失率的用户列表。通过用户调研推出一个最小可行化产品功能，并邀请种子用户进行试用。

- 用户方面：
  - 针对老年用户、无亲属、无伴侣用户的特征退出定制服务如亲属套餐、温暖套餐等，一方面加强与其它用户关联度，另一方对特定用户提供个性化服务。
- 服务方面：
  - 针对新注册用户，推送半年优惠如赠送消费券，以渡过用户流失高峰期。
  - 针对光纤用户和附加流媒体电视、电影服务用户，重点在于提升网络体验、增值服务体验，一方面推动技术部门提升网络指标，另一方面对用户承诺免费网络升级和赠送电视、电影等包月服务以提升用户黏性。
  - 针对在线安全、在线备份、设备保护、技术支持等增值服务，应重点对用户进行推广介绍，如首月/半年免费体验。
- 合同方面：
  - 针对单月合同用户，建议推出年合同付费折扣活动，将月合同用户转化为年合同用户，提高用户在网时长，以达到更高的用户留存。 
  - 针对采用电子支票支付用户，建议定向推送其它支付方式的优惠券，引导用户改变支付方式。