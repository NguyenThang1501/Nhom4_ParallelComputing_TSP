# Nhom4_ParallelComputing_TSP
Hướng dẫn cài đặt

Cài đặt Node.js

Link: [Download | Node.js (nodejs.org)](https://nodejs.org/en/download)

Clone source code từ github	

      git clone https://github.com/NguyenThang1501/Nhom4_ParallelComputing_TSP.git
              
Install các thư viện cần thiết
      
Di chuyển vào thư mục tsp, cài đặt npm


      cd tsp

      npm install

Numpy: 

      pip install numpy

Mpi4py: 

      pip install mpi4py

Flask: 

      pip install flask

Flask Cors: 

      pip install flask_cors

Chạy chương trình:

Trong thư mục backend, chạy lệnh: 

      python tsp_para.py

Mở một terminal khác, chuyển sang thư mục tsp, chạy lệnh: 

      npm start
      
  Lưu ý: Khi chạy lệnh: **`python tsp_para.py`** có thể gặp lỗi:

 ![image](https://github.com/NguyenThang1501/Nhom4_ParallelComputing_TSP/assets/109154036/57b96ad9-3f3e-448c-9930-b688ff7e00dd)

Khi đó, chúng ta cần chạy lệnh:

      pip install --upgrade watchdog

Và chạy lại:

      python tsp_para.py
