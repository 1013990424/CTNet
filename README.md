# CTNet: Color Transformation Network for Low-light Image Enhancement

# 1. Create Environment
Python 3.10.12  
Pytorch 2.6.0

# 2. Prepare Dataset
Download the following datasets:  
LOL and LOL-v2: https://github.com/flyywh/CVPR-2020-Semi-Low-Light  
SID: https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR  
SDSD: https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6

# 3. Testing

Testing with images   
the results can be download in 
https://pan.baidu.com/s/1JCMB6yB0hzvmlmrjGUFFuA?pwd=9n8t
code: 9n8t 

```python
# LOL
python test/LOL_img_test.py

# 4. Training
coming soon

# LOL-v2-real
python test/real_img_test.py

# LOL-v2-syn
python test/syn_img_test.py

# SDSD
python test/SDSD_img_test.py

# SID
python test/SID_img_test.py
```

Testing with pre-trained models  
```python
# LOL
python test/LOL_test.py

# LOL-v2-real
python test/real_test.py

# LOL-v2-syn
python test/syn_test.py

# SDSD
python test/SDSD_test.py

# SID
python test/SID_test.py

# unpair
python test/unpair_test.py 
```
