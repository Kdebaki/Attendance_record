CREATE DATABASE attendance_system;
USE attendance_system;

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL
);
SHOW TABLES;
SELECT * FROM attendance;
