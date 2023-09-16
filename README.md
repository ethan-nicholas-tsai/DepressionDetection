# Depression Detection on Online Social Network with Multivariate Time Series Feature of User Depressive Symptoms

The source code for [**Depression Detection on Online Social Network with Multivariate Time Series Feature of User Depressive Symptoms**](https://www.sciencedirect.com/science/article/pii/S0957417423000398) paper, accepted at ESWA 2023.  

## Abstract

In recent years, depression has attracted worldwide attention because of its prevalence and great risk for suicide. Existing studies have confirmed the feasibility of depression detection on online social networks. Most existing researches extract the overall features of users during a specific period, which cannot reflect the dynamic variation of depression. Besides, the methods proposed in these studies are often lack in interpretability and fail to establish the correlation between features and depressive symptoms in clinical. To address these problems, we propose a novel framework for depression detection based on multivariate time series feature of user depressive symptoms. Firstly, we construct and publish a well-labeled dataset collecting from the most popular Chinese social network platform Sina Weibo. To the best of our knowledge, it is the first large-scale depression dataset with complete collection of user tweeting histories, which includes 3,711 depressed users and 19,526 non-depressed users. Then, we propose a feature extraction method that reveals user depression symptoms variation in the form of multivariate time series. Moreover, we explore the various influencing factors to the performance of our proposed framework. In addition, we also explore the contributions of features to classification as well as their interpretability and conduct feature ablations on them. The experimental results show that our proposed method is effective and the extracted multivariate time series feature can well characterize the depressive state variation of users. Finally, we analyze the shortcomings and challenges of this study. Our research work also provides methods and ideas for tracking and visualizing the development of depression among online social network users.

![Illustration of our feature extraction method - DSTS](https://github.com/cyc21csri/DepressionDetection/blob/main/img/Method-DSTS.png)

## Dataset

You can download and aquire the information the datasets from the following links [SWDD](https://github.com/cyc21csri/SWDD).

### Requirements

Alongside the packages mentioned in the file "requirements.txt", you should install the following packages as below to run the **deep learning classifiers** in the paper.

- sktime-dl-0.2.0 (a modification to the origin package sktime-dl-0.1.0 by the author of this work)

- sktime-0.5.3 

We have packaged the two packages in zip files, see folder envs.

For the **machine learning classifiers** in this paper (when running the file 8_lab_length_ml.py, 9_lab_prop_ml.py, 10_lab_ablation_ml.py), however, **you should update the sktime package to version 0.8.1**.

## How to Run

We have uploaded all the source code needed to reproduce the results of experiments in the paper. Follow the instructions as below to run the code:

```
git clone https://github.com/cyc21csri/DepressionDetection.git
cd DepressionDetection
mkdir -p data/swdd dataset results
```

- Download `SWDD` data from [here](https://drive.google.com/file/d/1fNKtoo4SP98OAhalMjNRZfFqmQZsQ0fh/view?usp=sharing) and unzip it to `data/swdd` folder

- The source code is reorganized and the filename is renamed in the form of "[No.]\_[FileName]", where [No.] indicates the execution order of the scripts.

Notice that after you have executed the "4_make_ts_dataset.py", you should manually add the following info to the head of generated dataset file "train.ts" and "test.ts" in case it cannot be recognized as a time-series dataset.

```
@problemName MDDWeibo
@timeStamps false
@missing false
@univariate false
@dimensions 11
@equalLength true
@seriesLength 500
@classLabel true 0 1
@data
```

For more information of the time-series dataset format adopted in this paper, see [here](https://timeseriesclassification.com/) and download one of the dataset in [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) to obtain full comprehension of the format.

## Cite (BibTex)

Please cite the following paper, if you find our work useful in your research:

```
@article{cai2023depression,
  title={Depression Detection on Online Social Network with Multivariate Time Series Feature of User Depressive Symptoms},
  author={Cai, Yicheng and Wang, Haizhou and Ye, Huali and Jin, Yanwen and Gao, Wei},
  journal={Expert Systems with Applications},
  pages={119538},
  volume = {217},
  year={2023},
  doi={10.1016/j.eswa.2023.119538}
}
```

## Supplementary Infos to the paper

### Depression Symptom Descriptions

> in social media language

| #   | 症状名 (Symptom)                    | 症状描述 (Description)                                                                                                                                                                                                                                                                                                                |
|:---:|:--------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 1   | 悲伤情绪 (Sadness)                   | 长时间不开心、不高兴、不快乐，心情低落消沉、郁闷压抑、沮丧或绝望，经常总是悲伤想哭、伤心流泪、痛苦难过、感到空虚难熬惆怅 (Feeling unhappy, down, sad, or depressed for a long time, with a low, gloomy, distressed, discouraged or hopeless mood. Often feeling like crying, shedding tears of sorrow, agony and grief, feeling empty, unbearable melancholy and desolation.)                 |
| 2   | 兴致下降 (Loss of interest/pleasure) | 对几乎所有活动没兴趣、没意思没动力，乐趣明显减少、没有愉悦感，厌世、成天无精打采 (Having little interest or pleasure in almost all activities, feeling that things are meaningless or worthless, lacking motivation and drive. Experiencing a noticeable decrease in enjoyment, inability to feel joy, disenchantment with life, feeling lifeless and apathetic all day.) |
| 3   | 食欲问题 (Appetite problem)          | 食欲减退、经常饱、没胃口、想吐 (Loss of appetite, feeling full frequently, no desire to eat, feeling like vomiting.)                                                                                                                                                                                                                             |
| 4   | 睡眠障碍 (Insomnia)                  | 经常失眠睡不着、服用安眠药、熬夜到凌晨 (Frequently unable to sleep, insomnia, taking sleeping pills, staying up late into the early morning.)                                                                                                                                                                                                        |
| 5   | 急躁 (Agitation)                   | 精神性躁动、易感烦躁、坐立难安，言行冲动、易怒、易抓狂 (Mental agitation, easily irritated, restless, impulsive in words and actions, prone to anger, easily driven crazy.)                                                                                                                                                                                  |
| 6   | 精力不足 (Energy Loss)               | 经常感到累、困、昏晕乏力、疲惫没力气、没有精神 (Often feeling tired, sleepy, dizzy, weak, fatigued, lacking energy and vitality.)                                                                                                                                                                                                                        |
| 7   | 自责 (Self-blame)                  | 经常自我否定，我好没用、没有价值、一无是处、一事无成、好失败，让自己或家人失望，经常对不起、内疚自责、都是我的错 (Frequently self-negating, feeling useless, worthless, incompetent, a failure who lets myself or family down. Often feeling guilty, blaming and being hard on myself, thinking everything is my fault.)                                                                  |
| 8   | 注意力下降 (Concentration Problem)    | 注意力下降、无法专注、感到集中注意力困难、思考能力减退、犹豫不决、精神恍惚 (Decreased attention, inability to focus, difficulty concentrating, reduced thinking ability, indecisiveness, mental confusion.)                                                                                                                                                            |
| 9   | 自杀倾向 (Suicidal Ideation)         | 反复想到死亡、想死、自杀、结束生命，用刀片割腕自残、想跳楼自杀、计划自杀 (Repeated thoughts of death, wanting to die, suicide, ending one's life. Self-harming with razor blades, thinking of jumping off a building to commit suicide, making suicide plans.)                                                                                                        |
| 10  | 交感神经唤醒 (Sympathetic Arousal)     | 心慌、心悸、胸闷、喘不过气、颤抖、视力模糊、冒冷汗 (Feeling panic, heart palpitations, chest tightness, shortness of breath, trembling, blurred vision, breaking out in a cold sweat.)                                                                                                                                                                     |
| 11  | 恐慌 (Panic)                       | 经常好怕、害怕、恐惧、恐慌，想逃避 (Often feeling scared, afraid, terrified, panicked, wanting to escape.)                                                                                                                                                                                                                                         |

### Depressive search words

> To find depression indicative tweets in Sina Weibo. 
> 
> The crawler of Sina Weibo see https://github.com/cyc21csri/SinaWeiboCrawler

| #   | Chinese Search Words | English Meaning                                          |
|:---:|:--------------------:|:--------------------------------------------------------:|
| 1   | \#抑郁症\#              | Sina Weibo super topic of "Depression" in English        |
| 2   | 文拉法辛                 | "Venlafaxine" in English                                 |
| 3   | 舍曲林                  | "Sertraline" in English                                  |
| 4   | 度洛西汀                 | "Duloxetine" in English                                  |
| 5   | 抑郁 一无是处              | "Depression" and "Good for nothing" in English           |
| 6   | 抑郁 生无可恋              | "Depression" and "Don't want to live anymore" in English |
| 7   | 抑郁 没意思               | "Depression" and "Boring" in English                     |
| 8   | 抑郁 难熬                | "Depression" and "Suffering" in English                  |
| 9   | 抑郁 自残                | "Depression" and "Self-harm" in English                  |
| 10  | 抑郁 吃药                | "Depression" and "Take medicine" in English              |
| 11  | 抑郁 想哭                | "Depression" and "Want to cry" in English                |
| 12  | 抑郁 想死                | "Depression" and "Want to die" in English                |
