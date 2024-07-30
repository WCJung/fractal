**#처음 내려받을 때**

**1. 본인 로컬에 anaconda를 설치한다.**
   
**2. conda를 통해 가상환경을 설치한다. 아래 코드를 참고!** \
   code) \
   conda init \
   conda create -n (가상환경이름) python=3.10 \
   conda activate (가상환경이름)
   
**3. git을 설치하고, github ID와 email을 등록한다.** \
   code) \
   sudo apt-get install git    # Linux는 terminal에서 이걸 입력 \
   https://git-scm.com/download/win    # Windows는 여기서 다운받아 설치 \
   git --version    # version check; 버전나오면 설치OK \
   git config --global user.name (githubID) \
   git config --global user.email (email) \
   
**4. 이제 파일을 내려받는다! 그리고 packages를 설치한다.** \
   code) \
   git init
   git clone https://github.com/KY-HDC/fractal.git \
   pip install -r requirements.txt \

**5. 세팅 완료!**


**#코드수정하고 올리고 싶을 때(각자 branch로)**

**1. git clone해서 받은 코드를 필요에 맞게 수정한다.**

**2. 내 branch를 생성하고, 해당 branch로 이동한다.** \
   code) \
   git checkout -b vX.XO    # 현재 최신버전이 v4.6이므로, 여기서 수정이 생겼다면 v4.7A, v4.7B라고 명명한다. A, B는 작성자 이니셜임. \
   git status         # "On branch vX.XO"라고 나오면 OK \

**4. 내가 작성 또는 수정한 파일들을 올린다.** \
   code) \
   git add . \
   git commit -m "변경사항을 간략하게 적어줌" \
   git remote add origin https://github.com/KY-HDC/fractal.git \
   git push origin vX.XO \

**5. Github에서 본인 branch에 파일이 들어왔는지 확인한다.** 
