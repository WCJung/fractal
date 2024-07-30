* 처음 내려받을 때
1. 본인 로컬에 anaconda를 설치한다.
   
2. conda를 통해 가상환경을 설치한다. 아래 코드를 참고!
   eg)
   conda init
   conda create -n (가상환경이름) python=3.10
   conda activate (가상환경이름)
   
3. git을 설치하고, github ID와 email을 등록한다.
   eg)
   sudo apt-get install git    # Linux는 terminal에서 이걸 입력
   https://git-scm.com/download/win    # Windows는 여기서 다운받아 설치
   git --version    # version check; 버전나오면 설치OK
   git config --global user.name (githubID)
   git config --global user.email (email)
   
4. 이제 파일을 내려받는다! 그리고 packages를 설치한다.
   eg)
   git clone https://github.com/KY-HDC/fractal.git
   pip install -r requirements.txt

5. 세팅 완료!
