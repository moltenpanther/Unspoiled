# Unspoiled
AI-Powered Grocery Spoilage Prevention System, Senior Capstone

**Unspoiled** is an intelligent system designed to help grocery stores reduce product waste by identifying misplaced items that are likely to spoil. This Capstone project was developed as part of my Computer Science degree, where I served as the team lead and backend developer.

By combining **computer vision**, **custom logic**, a **database backend**, and a **web dashboard**, the system provides real-time alerts to store employees so they can take action before items go bad.

## Key Features

- 🛒 Identifies misplaced grocery items using AI-powered image analysis  
- 🔍 Flags spoilage risks based on product category and shelf location  
- 📊 Stores all detections and alerts in a structured relational database  
- 🌐 Offers a clean, responsive web interface for employee use  
- 🔔 Helps reduce food waste, saving both inventory and cost  

## How It Works

- Images from store shelves are analyzed using a trained **computer vision model** built with TensorFlow, YOLO (You Only Look Once), and OpenCV  
- The system detects items and compares their current placement with expected locations  
- Misplaced items (e.g., frozen goods left in non-refrigerated aisles) are flagged for review  
- Flagged data is stored in a **PostgreSQL database** and displayed in a **Vue-based web dashboard**  
- Employees can access the interface to quickly respond to alerts and prevent spoilage

## Technologies Used

- **Python** for backend logic  
- **TensorFlow** for AI model training and inference  
- **OpenCV** for image processing  
- **Spring Boot** for the API  
- **PostgreSQL** for storing detection results  
- **HTML / CSS / JavaScript** for the frontend

## My Contributions

As the **project lead**, I was responsible for:

- Designing and training the AI model  
- Creating an auto-labeling algorithm to speed up the AI training
- Writing the item placement detection logic  
- Leading our team’s planning, testing, and integration efforts

## Possible Future Improvements

- Support for real-time video input  
- Mobile notifications for in-store use  
- Integration with real-world store inventory systems  
- Improved accuracy with larger and more diverse datasets

## License

This project is released under the MIT License.
