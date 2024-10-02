#ifndef DISTANCE_H
#define DISTANCE_H

#include <cmath>
#include <geometry_msgs/Point.h>  

inline double euclideanDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

#endif // DISTANCE_H
