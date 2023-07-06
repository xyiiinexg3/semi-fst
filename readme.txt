conda create -n fst python=3.8
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install transformers==3.0.2
pip install nltk nlpaug pyskiplist statistics fitlog
----------
conda install libpython m2w64-toolchain -c msys2 失败
pip install kenlm 失败
pip install https://github.com/kpu/kenlm/archive/master.zip 死慢 半个小时有的
error: Microsoft Visual C++ 14.0 or greater is required 安装Microsoft C++ Build Tools https://zhuanlan.zhihu.com/p/165008313 没用
pip install https://github.com/kpu/kenlm/archive/master.zip 一样报错
visualcppbuildtools_full.exe 包损失
pip install pypi-kenlm 报错
git bash下 pip install -e git+https://github.com/kpu/kenlm.git#egg=kenlm
安装Vsstudio https://blog.csdn.net/stay_foolish12/article/details/118085599


