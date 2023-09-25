# Project - Parallel Algorithms

## Objective 

This project implements parallel algorithms using CUDA C. 



## TODO

- [ ] Backend 
    - [ ] #TODO: #2 API to connect frontend to CUDA image processing functions, e.g., colorToGrayscale, imageBlur
- [ ] Frontend
    - [ ] #TODO:
- [ ] CUDA Parallel Processing
    - [ ] #TODO: #3 function: color to grayscale
    - [ ] #TODO: #4 function: image blurring
- [ ] API  
    - [x] #TODO: #1 Write basic hello world api & test it using curl
- [ ] Unit Testing 
    - [ ] #TODO: #5 Add robust testing in every part of this project
    - [ ] #TODO: #6 Use GitHub Actions (CI / CD ) to incorporate testing 




## Folder Structure

```
project_root/
│
├── api/               (API-related files)
│   ├── app.py         (Flask application)
│   ├── requirements.txt (Python dependencies)
│   ├── routes/        (API route handlers)
│   └── ...
│
├── cuda/              (CUDA C/C++ code)
│   ├── kernel1.cu     (CUDA kernel source files)
│   ├── kernel2.cu
│   ├── include/       (CUDA kernel headers)
│   └── ...
│
├── frontend/           (Frontend code - could be React, Angular, Vue, etc.)
│   ├── src/            (Frontend source code)
│   ├── public/         (Static files)
│   ├── package.json    (Frontend dependencies)
│   ├── webpack.config.js (Webpack configuration if using)
│   └── ...
│
├── static/             (Static assets for the frontend, e.g., images, CSS)
│
├── templates/          (HTML templates for your Flask app, if applicable)
│
├── data/               (Input or reference data for image processing)
│
├── docs/               (Project documentation and README files)
│
├── tests/              (Unit tests and integration tests)
│
├── .gitignore          (Git ignore file)
│
└── README.md           (Project README and documentation)

```