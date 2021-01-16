

#  淘宝用户业务行为分析

[TOC]

### 项目简介

淘宝网是亚太地区较大的网络零售、商圈，由阿里巴巴集团在2003年5月创立。淘宝网是中国深受欢迎的网购零售平台，拥有近5亿的注册用户数，每天有超过6000万的固定访客，同时每天的在线商品数已经超过了8亿件，平均每分钟售出4.8万件商品。

本项目数据集包含淘宝平台的2017年11月25日至2017年12月3日之间，有行为的约一百万随机用户的所有行为（行为包括点击、购买、加购、喜欢）。数据集的每一行表示一条用户行为，由用户ID、商品ID、商品类目ID、行为类型和时间戳组成，并以逗号分隔。用于研究用户隐式反馈问题。

用户历史行为数据从类型上主要分为显式反馈数据和隐式反馈数据。其中显式反馈数据是用户主动进行评价，例如评分、评级、评论等，能够明确地表达用户喜好的数据；隐式反馈数据则利用用户点击行为、用户浏览记录、购买历史等信息，不能直接表达用户喜欢与不喜欢，只是展现用户兴趣的数据。用户评分等显式反馈虽然能够明显表达用户喜好，但受很多因素影响且不易获取，因此相较于显式反馈，隐式反馈更易获取。隐式反馈存在于互联网上，它是用户在使用服务过程中留下的数据。用户在进行浏览新闻、购物、听音乐等行为中的各种选择，都可作为隐式反馈。隐式反馈数据的收集范围广、成本低、不易引起用户反感，但数据规模一般比较大。隐式反馈虽然能够隐式地表达用户的兴趣和喜好，即能够表达出用户的正倾向，却不易表达用户的负倾向，同时隐式反馈数据通常伴有噪声。

本项目针对淘宝app的运营数据，以行业常见指标对用户行为进行分析，包括UV、PV、新增用户分析、漏斗流失分析、留存分析、用户价值分析、复购分析等内容。

- 数据集：[阿里云天池](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1)

- 工具：Excel、SQL、Navicat、Python、Tableau

- 按照数据分析的标准流程进行本次的数据分析:

  1. 明确问题： 明确数据分析的真是目的

  2. 理解数据： 据获取和数据探索

  3. 数据清洗： 数据降噪处理，使数据成为专家数据

  4. 数据分析和可视化：分析并可视化结果

  5. 结论和建议： 结果解读，得出有价值的结论且提出相关建议

### 1、明确问题

#### 1.1 分析目的

淘宝数据分析的目的是把隐藏在一大堆看起来杂乱无章的数据背后的信息提炼出来，总结它们的原因或者规律
等。通过对数据集中前500万行数据的分析，提炼用户隐式反馈信息，发现用户行为规律模式，为用户提供更高
精度的产品运营策略，提升用户粘度。

#### 1.2 分析思路

用户的隐式反馈研究，主要是根据用户的行为数据进行分析，所以，也就是“人-货-场”中的“**人**”的数据。

通过对淘宝用户行为数据的分析，为以下问题提供解释和改进建议：

1）基于AARRR漏斗模型，使用常见电商分析指标，从新增用户数量、各环节转化率、新用户留存率三个方面进行分析，确定影响新增用户数量的因素，找到需要改进的转化环节，发现留存现存问题

2）研究用户在不同时间尺度下的行为规律，找到用户在不同时间周期下的活跃规律

3）找出最具价值的核心付费用户群，对这部分用户的行为进行分析

4）找到用户对不同种类商品的偏好，制定针对不同商品的营销策略

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjsjck96tj30qc0mcaco.jpg" style="zoom:50%;" />

根据用户指标体系以及数据情况，可分析用户流量数量、流量质量、用户质量等方面。

#### 1.3 提出问题

总结出以下具体问题：

用户行为转化分析：分析转化率情况，提出改善建议。

用户行为习惯分析：以PV和时间数据找到出用户活跃规律，从而进行精准营销。

用户类目偏好分析：根据商品的浏览，收藏，加购，购买的指标，探索用户偏好商品，优化推荐。

用户价值分析：用户价值分层，针对不同用户提出不同的运营策略。

### 2、数据理解

#### 2.1 数据详情

​	数据集包含了2017年11月25日至2017年12月3日之间，约一百万随机用户的所有行为（行为包括点击、购买、加购、喜欢）。数据集大小情况为：用户数量约100万（987,994），商品数量约410万（4,162,024），商品类目数量9,439以及总的淘宝用户行为记录数量为1亿条（100,150,807）。数据整体情况参考如下表格。

|    列名    |                      解释                       |
| :--------: | :---------------------------------------------: |
|   用户ID   |           整数类型，序列化后的用户ID            |
|   商品ID   |           整数类型，序列化后的商品ID            |
| 商品类目ID |         整数类型，序列化后的商品类目ID          |
|  行为类型  | 字符串，枚举类型，包括('pv','buy','cart','fav') |
|   时间戳   |                行为发生的时间戳                 |

其中，用户行为有四种：

| 行为类型 |            说明            |
| :------: | :------------------------: |
|    pv    | 商品详情页，相当于点击商品 |
|   buy    |          购买商品          |
|   cart   |       商品加入购物车       |
|   fav    |          收藏商品          |

#### 2.2 数据预处理

```python
# 导入相关的库
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import squarify
import warnings
import gc
from datetime import datetime,timedelta
%matplotlib inline
warnings.filterwarnings('ignore')
```

```python
# 原始数据超过1亿行，抽取前500万行的数据进行处理
df = pd.read_csv('/Users/fq/Desktop/UserBehavior.csv',nrows= 5000000)
df.head(5)
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjsx95wlrj30f4098aat.jpg" style="zoom:50%;" />

```python
#列名命名
df.columns = ['user_id','product_id','item_id','behaviour_type','timestamp']
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjt5eh21mj30me094my2.jpg" style="zoom:50%;" />

```python
#查看数据信息：
df.info()
df.head()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjt9pem3rj30jo0bajsu.jpg" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjtau8tapj30ma08wgmi.jpg" style="zoom:50%;" />

### 3、数据清洗

#### 3.1 去除重复数据

```python
# 数据清洗 
# 计算重复值，并删除：
print('重复数据有：',df.duplicated().sum())   # 重复数据有5个
df = df.drop_duplicates()
print('删除后，重复数据有：',df.duplicated().sum())  #删除重复值
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjtd0sgtaj309o0200sq.jpg" style="zoom:50%;" />

**重复值为5条，删除重复值后，数据集变为4999995行**。

#### 3.2 查看缺失值

```python
# 查看各字段的缺失值数量：
df.isnull().sum()   # 无缺失字段
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjtf29lajj309y05omxe.jpg" style="zoom:50%;" />

**数据无缺失值。**

#### 3.3 日期与时间的处理

时间的标准格式转换：

```python
#原始数据的时间格式是英国时区，换成中国东八区需要+8小时
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')+timedelta(hours = 8)
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.hour
df.head()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjthuwer3j30ve08ygmv.jpg" alt="" style="zoom:50%;" />

#### 3.4 异常值处理

1 找出日期小于2017-11-25日的值，并删除：

```python
df[df['timestamp']<datetime(2017,11,25)].index      
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjtqvgqcwj30za05maaz.jpg" style="zoom:50%;" />

小于日期小于2017-11-25日的值为2600行，

```python
df.drop(df[df['time']<datetime(2017,11,25)].index,inplace=True)
```

删除后行数为：4997395

2 找出日期大于2017-12-04日的值，并删除：

```python
df[df['timestamp']>datetime(2017,12,4)].time.count()      
```

日期大于2017-12-04日的值为32行

```python
df.drop(df[df['timestamp']>datetime(2017,12,4)].index,inplace=True)
df.info()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjyn5x5quj30l40d6762.jpg" style="zoom:50%;" />

删除后行数为：4997363

### 4、数据分析和可视化

#### 4.1 数据整体情况

##### 4.1.1  访问用户总数UV、页面总访问量PV、人均浏览次数、成交量

```python
UV = df['user_id'].unique().size
PV = df[df['behaviour_type'] == 'pv'].user_id.count()
user_per = PV/UV
buy_count = df[df['behaviour_type'] == 'buy'].user_id.count()
print('计算各值为：',UV,PV,user_per,buy_count)
```

计算得出各值为： 48984   4472599   91.30734525559366   100126

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjyrt3pxgj30jo03cq34.jpg" style="zoom:50%;" />

##### 4.1.2 日均UV、PV、人均浏览次数、成交量

```python
st_perday = df[['user_id','date']].groupby('date').count()
st_perday['pv_perday'] = df[df['behaviour_type'] == 'pv'][['behaviour_type','date']].groupby('date').count()
st_perday['deep_perday'] = st_perday['user_id']/st_perday['pv_perday']
st_perday['buy_perday'] = df[df['behaviour_type'] == 'buy'][['behaviour_type','date']].groupby('date').count()
st_perday.columns = ['日访客数','日浏览数','日人均浏览数','日成交量']
st_perday.index.name = '日期'
st_perday
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjyt3dv2hj30je0gs764.jpg" style="zoom:50%;" />

绘制日访客数、日浏览数以及日成交量趋势图：

```python
# 解决中文显示问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.figure(figsize=(12,3))
plt.plot(st_perday[['日访客数','日浏览数']])
plt.legend(labels = ['日访客数','日浏览数'],loc = 2)
plt.twinx()
plt.plot(st_perday[['日成交量']],color = 'black')
plt.legend(['日成交量'],loc =4)
plt.title('日访客数、日浏览数以及日成交量趋势图')
plt.show()
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmjz116l9lj33c00u0n8m.jpg)

12月2,3号是周六日，其数值同比上周增长幅度较大，应该是双12活动开始预热。

##### 4.1.3 用户的复购率和跳失率

```python
buy_count = df[df['behaviour_type'] == 'buy'].user_id.count()
a = df[df['behaviour_type'] == 'buy']['user_id'].value_counts()>=2
re_buy = a[a].size
# 计算复购率：
re_buy_rate = (re_buy / buy_count) *100
print('复购次数：%i\n实际购买次数：%i\n复购率：%.2f%%'%(re_buy,buy_count,re_buy_rate))
```

复购次数：21892
实际购买次数：100126
复购率：21.86%

数量级为10万的情况下，复购率为65.87%，但是在数量级为5百万的情况下复购为仅为21.92%。数量级是非常重要的，在可能的情况还是有分析尽可能多的数据。**复购率**反映的是用户的粘性，忠诚度，不同的复购率会采取不用运营策略。复购率如果小于**15%**，运营策略应该重点放在新客获取上，复购率超过**60%**则应该放在忠诚客户维系上，处在中间则应采用混合模式。

```python
# 计算跳失率：
a = df[df['behaviour_type']=='pv']['user_id'].value_counts()==1
a = a[a].size/UV*100
print('跳失率为：%0.5f%%'%a)
```

跳失率为：0.06941%

该数据量的数据最终跳失率相对非常低。

#### 4.2 用户行为转化分析

##### 4.2.1 基于行为转化漏斗分析

计算用户行为分布的数量：

```python
be_value = df[['user_id','behaviour_type']].groupby('behaviour_type').count().sort_values(by = 'user_id')
be_value.index.name='行为分布值'
be_value
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjz5iy18ej307c0943yp.jpg" style="zoom:50%;" />

绘制用户行为分布条形图：

```python
y = be_value['user_id']
x = ['用户购买数','用户加购数','用户收藏数','用户浏览数']
plt.barh(x,y)
plt.title("用户行为分布情况")
plt.savefig('2.png',dpi = 600)
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzcpu4voj31900u0jun.jpg" style="zoom:30%;" />

浏览后的加购的转化率为6.24%

浏览后的收藏的转化率为3.24%

浏览后的购买的转化率为2.23%

可以看出用户在浏览后的行为转化率都较低，那么用户转化是否也是如此？

##### 4.2.2 基于用户转化漏斗分析

每种行为的用户数量（去除重复值）

```python
pv_user = df[df['behaviour_type'] == 'pv']['user_id'].unique().size
cart_user = df[df['behaviour_type'] == 'cart']['user_id'].unique().size
fav_user = df[df['behaviour_type'] == 'fav']['user_id'].unique().size
buy_user = df[df['behaviour_type'] == 'buy']['user_id'].unique().size
be_user = {'浏览用户数':pv_user,'加购用户数':cart_user,'收藏用户数':fav_user,'购买用户数':buy_user}
be_user = pd.Series(be_user)
be_user
```

浏览用户数    48782
加购用户数    36906
收藏用户数    19489
购买用户数    33286

```python
be_user.plot(kind='barh')
plt.title("用户分布情况")
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzj34l00j31900u0diw.jpg" style="zoom:33%;" />

用户浏览后加购的转化率为75.49%

用户浏览后收藏的转化率为39.77%

用户浏览后购买的转化率为68.03%

由上图可以看出，用户并未在点击后就大量流失,而且有高达68.03%的付费转化率。结合业务场景分析，行为转化率较低的原因是用户要对同种产品进行比较，以及单纯的浏览商品。还有为什么收藏用户数远小于加购的用户数？结合业务场景，推测可能因为收藏后不可以直接购买，但是加购后可以直接购买。可以引导用户将单纯的收藏行为，转化为收藏并加购，来提升用户购买转化率。

#### 4.3 用户行为习惯分析

##### 4.3.1 整体用户行为活跃情况

计算整体用户行为按小时分布的数量：

```python
time_browse = df.groupby(['time','behaviour_type'])['user_id'].count().unstack()
time_browse.columns = ['购买','加购','收藏','浏览']
time_browse.index.name = '小时'
time_browse.head()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzl1o8w4j30h00d0gma.jpg" style="zoom:50%;" />

绘制整体用户行为活跃情况折线图：

```python
plt.figure(figsize=(15,3))
time_browse[['加购','收藏','购买']].plot(style = '-o')
plt.legend(loc = 2)
plt.title('用户行为习惯按时间分布图')
plt.xlabel('小时')
plt.twinx()
time_browse['浏览'].plot(kind = 'bar',figsize=(15,3),facecolor = 'orange',alpha = 0.5)
plt.legend(loc=4)
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzn628jhj34600u0h02.jpg" style="zoom:50%;" />

由上图可知，每日0点至3点用户活跃度迅速下降，降到最低值，4点到10点用户活跃度开始上升，10点至18点用户活跃度较平稳，18点后用户活跃度开始快速上升，并在20-22时达到一天中用户活跃度的最高值，符合人群的作息规律。根据用户不同活跃情况制定不同的运营策略，在用户活跃高峰期增加广告以及运营手段的投入，达到更高用户转化率。

#### 4.4 用户类目偏好分析

##### 4.4.1 计算本次分析的产品类目数量

```python
print('本次分析的产品类目数量为：%i \n商品数量为：%i'%(df['category_id'].unique().size,df['item_id'].unique().size))
```

本次分析的产品类目数量为：7352

商品数量为：1080286

对所有商品购买次数进行排名：

```python
# 商品购买次数排名：
buy_count = df[df['behaviour_type'] == 'buy'].groupby('item_id')['user_id'].count()
buy_count = buy_count.value_counts(sort = True)[:10]
buy_count
buy_count.plot(kind = 'bar')
plt.title('商品购买次数分析')
plt.xlabel('购买次数')
plt.show()
```

取所有商品购买前十的商品绘图展示：

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzr6y51hj31900u0781.jpg" style="zoom:33%;" />

##### 4.4.2 不同行为下的排名前20的商品

浏览排名前20的商品:

```python
be_pv = df[df['behaviour_type'] == 'pv'].groupby('item_id')['user_id'].count().sort_values(ascending = False)[:20]
plt.figure(figsize=(12,3))
squarify.plot(sizes = be_pv.values,label=be_pv.index,value=be_pv.values,alpha = 0.5,edgecolor = 'white')
# 除去坐标轴
plt.axis('off')
# 除上边框和右边框刻度
plt.tick_params(top = 'off', right = 'off')
plt.title('浏览排名前20的商品分布情况')
plt.savefig('3.png',dpi = 600)
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzufmgb4j33c00u0drg.jpg" style="zoom:50%;" />

加购排名前20的商品:

```python
be_cart = df[df['behaviour_type'] == 'cart'].groupby('item_id')['user_id'].count().sort_values(ascending = False)[:20]
plt.figure(figsize=(12,3))
squarify.plot(sizes = be_cart.values,label=be_cart.index,value=be_cart.values,alpha = 0.5,edgecolor = 'white')
# 除去坐标轴
plt.axis('off')
# 除上边框和右边框刻度
plt.tick_params(top = 'off', right = 'off')
plt.title('加购排名前20的商品分布情况')
plt.show()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmjzwil059j33c00u0wop.jpg" style="zoom:50%;" />

收藏排名前20的商品:

```python
be_fav = df[df['behaviour_type'] == 'fav'].groupby('item_id')['user_id'].count().sort_values(ascending = False)[:20]
plt.figure(figsize=(12,3))
squarify.plot(sizes = be_fav.values,label=be_fav.index,value=be_fav.values,alpha = 0.5,edgecolor = 'white')
# 除去坐标轴
plt.axis('off')
# 除上边框和右边框刻度
plt.tick_params(top = 'off', right = 'off')
plt.title('收藏排名前20的商品分布情况')
plt.savefig('4.png',dpi = 600)
plt.show()
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmjzyejax0j33c00u07f0.jpg)

购买排名前20的商品:

```python
be_buy = df[df['behaviour_type'] == 'buy'].groupby('item_id')['user_id'].count().sort_values(ascending = False)[:20]
plt.figure(figsize=(12,3))
squarify.plot(sizes = be_buy.values,label=be_buy.index,value=be_buy.values,alpha = 0.5,edgecolor = 'white')
# 除去坐标轴
plt.axis('off')
# 除上边框和右边框刻度
plt.tick_params(top = 'off', right = 'off')
plt.title('购买排名前20的商品分布情况')
plt.savefig('5.png',dpi = 600)
plt.show()
```

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmjzzgucvmj33c00u012v.jpg)

```python
print('浏览排名前20的商品有%i件是加购，%i件收藏，%i件购买 \n加购排名前20的商品有%i件是购买的\n收藏排名前20的商品有%i件是购买的'
%(be_pv[be_pv.index.isin(be_cart.index)].count(),
  be_pv[be_pv.index.isin(be_fav.index)].count(),
  be_pv[be_pv.index.isin(be_buy.index)].count(),
  be_cart[be_cart.index.isin(be_buy.index)].count(),
  be_fav[be_fav.index.isin(be_buy.index)].count()
))
```

浏览排名前20的商品有10件是加购，10件收藏，2件购买 
加购排名前20的商品有5件是购买的
收藏排名前20的商品有1件是购买的

可以看出浏览最多的商品和购买最多的商品差别略大，最吸引用的产品却没有很好的转化成销量，应调整相关商品的运营策略，减少差距。同时，可以看出收藏的行为距离购买的行为较远，更应鼓励用户多进行加购行为。

#### 4.5 用户价值分析

用户RFM分层理论： RFM是3个指标的缩写，最近一次消费时间间隔（Recency），消费频率（Frequency），消费金额（Monetary）。而RFM模型就是通过这三项指标，来描述客户的价值状况，从而得到分群的客户。其中

R是指用户的最近一次消费时间距现在有多长时间了，这个指标反映用户流失与复购（粘性）。

F是指用户在指定观察的周期内消费了几次。这个指标反映了用户的消费活跃度（忠诚度）。

M是指用户在指定的观察周期内在平台花了多少钱，这个指标反映用户对公司贡献的价值（营收）。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmk31um4shj30vg0eyn6l.jpg" style="zoom:50%;" />

由于数据集的统计的时间仅有9天，而且没有消费数据，只能根据消费频率以及消费时间进行分析，规则设置如下：

用户价值评分表：

| 按价值打分 | 最近一次消费间隔（R值） |   消费频次（F值）   |
| :--------: | :---------------------: | :-----------------: |
|     1      |        间隔>7天         | 购买次数在1-10之间  |
|     2      |       间隔在5-7天       | 购买次数在10-20之间 |
|     3      |       间隔在3-4天       | 购买次数在20-30之间 |
|     4      |       间隔在0-2天       |    购买次数>30次    |

用户价值分析表：

|     按价值打分      | R值：3，4 （高） | R值：1，2（低） |
| :-----------------: | :--------------: | :-------------: |
| **F值：3，4（高）** |   重要价值客户   |  重要保存客户   |
| **F值：1，2（低）** |   重要发展客户   |  重要挽留客户   |

##### 4.5.1 用户价值指标计算

```python

# 购买时间间隔：
R_day = (datetime(2017,12,4) - df[df['behaviour_type'] == 'buy']['timestamp']).dt.days
a = df[df.index.isin(R_day.index)].user_id
a=pd.DataFrame(list(zip(a, R_day)))
a.drop_duplicates()
a = a.groupby(0).min()
a['id'] = a.index
a = a.sort_values(by = 'id')
# 购买频率
buy_count = df[df['behaviour_type'] == 'buy'].user_id.value_counts().sort_index()
user_value = pd.DataFrame(list(zip(buy_count,a[1])),index=buy_count.index)
user_value.columns = ['消费频次','购买时间间隔']
user_value.index.set_names('用户ID',inplace = True)
print('用户消费频次，购买时间间隔统计表')
user_value
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmlw3kt6t0j30e00o20tw.jpg" style="zoom:50%;" />

##### 4.5.2 计算价值分数

```python
#计算价值分数
user_value['R值'] = pd.cut(user_value['购买时间间隔'],bins= [-1,2,4,7,9],labels = [4,3,2,1])
user_value['F值'] = pd.cut(user_value['消费频次'],bins =[0,10,20,30,90],labels=[1,2,3,4])
#将分类数据转化为数值类型
user_value[['R值','F值']] = user_value[['R值','F值']].apply(pd.to_numeric)
print('用户价值分层表')
user_value
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmlwef7gz3j30gc0o4t9x.jpg" style="zoom:50%;" />

##### 4.5.3 分层客户计算

```python
#客户分层函数
def customer_type(frame):
    customer_type = []
    for i in range(len(frame)):
        if frame.iloc[i,2]>=3 and frame.iloc[i,3]>=3:
            customer_type.append('重要价值客户')
        elif frame.iloc[i,2]>=1 and frame.iloc[i,3]>=3:
            customer_type.append('重要保存客户')
        elif frame.iloc[i,2]>=3 and frame.iloc[i,3]>=1:
            customer_type.append('重要发展客户')
        elif frame.iloc[i,2]>=1 and frame.iloc[i,3]>=1:
            customer_type.append('重要挽留客户')
    frame['classification'] = customer_type
customer_type(user_value)
user_value
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmlwjp90eej30na0mcdhy.jpg" style="zoom:50%;" />

```python
print('客户分类统计值：')
user_value.classification.value_counts()
```

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gmlwj7gqy4j30ha06ut9d.jpg" style="zoom:50%;" />

总结：可以看出，大部分客户的上次消费时间距离12.4日都大于5天，可能是都在为双十二做准备，以

及双十一大促后的消费减少。

|   用户类型   | 最近一次消费时间间隔（R值） | 消费频率（F值） | 消费金额（M值） |                        营销策略                        |
| :----------: | :-------------------------: | :-------------: | :-------------: | :----------------------------------------------------: |
| 重要价值用户 |             高              |       高        |       高        |     最优质用户，应该提高用户满意度，增加用户留存率     |
| 重要发展用户 |             低              |       高        |       高        |    购买频率低，可以增加用户个性化推荐，增加购买频率    |
| 重要保存用户 |             高              |       低        |       高        |   消费间隔大，增加个性化推荐内容及次数，以免用户流失   |
| 重要挽留用户 |             低              |       低        |       高        |           潜在有价值用户，了解原因，进行挽留           |
|   潜力用户   |             高              |       高        |       低        | 忠诚用户，但累计消费金额低，可以适当进行引导，消费升级 |
|    新用户    |             低              |       高        |       低        |       新用户，可以利用低门槛优惠券等方式吸引消费       |
| 一般维持用户 |             高              |       低        |       低        |                   需分析进行运营激活                   |
|   流失用户   |             低              |       低        |       低        |        相当于流失用户，可进行适当推送，重新激活        |

### 5、分析与建议

#### 5.1 用户行为的转化分析

​	通过对事件以及用户的转化分析，可以看出用户从点击到购买的转化率是很高的68.03%。而浏览后购买的转化率仅有2.23%，故从浏览到购买的行为转化是一个提高的重点。针对这一环节的建议有：

- 优化电商平台的搜索匹配度和推荐策略，提高筛选精确度，并对搜索和筛选的结果排序的优先级进行优化；
- 可以给客户提供同类产品比较的功能，让用户不需要多次返回搜索结果进行点击查看，方便用户确定心仪产
  品，增加点击到后续行为的转化；
- 优化收藏到购买的操作过程，增加用户收藏并加购的频率，以提高购买转化率。

#### 5.2 用户行为习惯分析

​	可以看出用户的活跃时间高峰期主要在20-22点，此时使用人数最多，活动最容易触达用户，所以可以将营销
活动安排在这个时间段内，来进行引流并转化。

​	在研究的9天内共有两个周末，第一个周末用户活跃度并没有明显变化，而第二个周末推测是因为有双十二的预热，导致用户点击和加购出现明显增加。故可以扩大研究时间范围对推测进行验证，若推测正确，则可以通过在周末推出营销活动来挖掘用户周末购物的欲望。

#### 5.3 用户类目偏好分析

可以看出商品销量主要是依靠长尾效应，而非爆款商品的带动。但是通过对商品品类的分析可以看出能吸引用户注意力的商品购买转化率并不高，是一个提高销量的突破口。针对用户关注度高但销量不高的这部分产品，可以从以下几个方面着手：

- 商家在详情页上的展示突出用户重点关注的信息，优化信息的呈现方式，减少用户的时间成本；
- 增加这些产品的质量管控力度，加强对用户反馈建议如评论的管理，认真采纳并根据自身的优劣势进行商品优化。
- 此外，对于购买top20的商品，可以在电商首页对这些品类的商品优先进行展现，以满足用户的购买需求。

#### 5.4 用户价值分析

​	可以对不同的用户群体采用**不同的管理策略**，达到对不同的客户群进行精准营销的目的：

- 对于重要价值用户，需要重点关注并保持， 应该提高满意度，增加留存；
- 对于重要发展客户和重要保持用户，可以适当给点折扣或捆绑销售来增加用户的购买频率；
- 对于重要挽留客户，需要关注他们的购物习性做精准化营销，以唤醒他们的购买意愿。

