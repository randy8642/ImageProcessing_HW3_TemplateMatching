# ImageProcessing_HW1_TemplateMatching

## 作業說明
利用上課學到的 Texture Matching 或是 Shape Matching，學習提供的兩組 data 中的 template image，並在對應的影像中，將 matching 的結果顯示出來。

- 可使用 openCV 處理開檔以及顯示結果。
- matching 的主體演算法部份（包含學習 template 以及實際 matching）需要自己撰寫。
- 利用上課提到的加速的方法，有額外加分。
- 結果的部份須顯示： match result 的中心點座標以及相似度（如100-MatchResult.jpg與Die-MatchResult.jpg所示）。

繳交作業的檔案須包含 （將所有檔案壓縮成 zip 檔，檔名為 學號_姓名_HW1.zip）：
1. 程式碼 source code
2. 報告，內容包含：
   1. 開發時遇到的問題、解決方法、match 一張影像所需時間。
   2. 跟 openCV 的結果做 benchmark ，速度有比較快有加分。

處理範例圖
![](/sample/100-MatchResult.jpg)
![](/sample/Die-MatchResult.jpg)

## 功能實現
### RGB to Gray(single channel)
```python
def convertBGR2GRAY(img):
   res = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
   res = res.astype(np.float32)
   return res
```

### Downsampling 
```python
def createSubsampleImgs(originImage, originTemplate):
   sample_imgs = [originImage]
   sample_templates = [originTemplate]

   # 取影像的奇數行與列來縮小影像大小
   for _ in range(3):
      sample_imgs.append(sample_imgs[-1][::2, ::2])
      sample_templates.append(sample_templates[-1][::2, ::2])
   
   return sample_imgs, sample_templates
```

### Template Matching
使用Normalized Correlation Coefficient計算圖片與模板的相似度
```python
def templateMatching(img, temp):
   # 將圖片依照模板大小重新排列/裁切
   sub_matrices = change2ConvView(img, temp)

   # 取得模板的長寬
   template_h, template_w = temp.shape

   # 計算normalized Template
   T_norm = temp - np.mean(temp)
   # 計算normalized mean Image  
   I_norm_mean = np.einsum('klij->kl', sub_matrices) / (template_w*template_h)

   # 分子部分
   m = np.einsum('ij,klij->kl', T_norm, sub_matrices)
   n = I_norm_mean * np.sum(T_norm)

   # 分母部分
   T_norm_sum = np.sum(T_norm**2)   
   I_norm_sum = np.einsum('klij,klij->kl', sub_matrices, sub_matrices) \
                  - 2 * np.einsum('klij->kl', sub_matrices) * I_norm_mean \
                  + (template_w*template_h) * I_norm_mean**2

   # 結果
   R = (m - n) / np.sqrt(T_norm_sum * I_norm_sum)

   return R
```

### de-duplicate
```python
def deduplicate(location: np.ndarray, similarity: np.ndarray, threshold):
   results_location = []
   results_sim = []

   while len(location) > 0:    
      p = location[0]        

      # 計算p點到其他目標點的距離
      distance = np.sqrt(np.sum((location - p)**2, axis=1))
      # 找出距離過近的點
      nearPoint = np.argwhere(distance < threshold).flatten()

      # 找出這些點之中相似度最高的點
      sim_argmax = np.argmax(similarity[nearPoint])
      # 保留該點
      results_location.append(location[nearPoint][sim_argmax])
      results_sim.append(similarity[nearPoint][sim_argmax])
      # 刪除其他點
      location = np.delete(location, nearPoint, axis=0)
      similarity = np.delete(similarity, nearPoint, axis=0) 
   
   return np.array(results_location), np.array(results_sim)
```


## 處理過程說明
### Step 1 : 轉為單通道影像

### Step 2 : Downsampling Image and Template

### Step 3 : 對最小的圖片執行Template Matching
```python

```

### Step 4 : 使用目標點對較大圖片裁切
```python

```

### Step 5 : 在裁切後的較大圖片執行Template Matching
```python

```

### Step 6 : 在原圖繪製目標框
```python

```

### Step 7 : 儲存圖片
```python

```

## 處理結果
### 樣張1
![](/source/100-1.jpg)
![](/result/100-1.jpg)
### 樣張2
![](/source/100-2.jpg)
![](/result/100-2.jpg)
### 樣張3
![](/source/100-3.jpg)
![](/result/100-3.jpg)
### 樣張4
![](/source/100-4.jpg)
![](/result/100-4.jpg)
### 樣張5
![](/source/Die1.jpg)
![](/result/Die1.jpg)
### 樣張6
![](/source/Die2.jpg)
![](/result/Die2.jpg)

## 與OpenCV比較
使用`100-1.jpg`/`100-21.jpg`/`100-3.jpg`/`100-4.jpg`共4張圖重複執行5次的平均結果比較
| method          |  costTime   |
|:----------------|------------:|
| OpenCV function |  0.113902 s |
| Self-developed  |  0.236849 s |
