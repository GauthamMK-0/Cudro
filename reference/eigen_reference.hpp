#pragma once

#include <Eigen/Dense>
#include <vector>

namespace cudro::reference {

// Independent Planar 2R Reference Model
class Planar2RReference {
public:
    // Computes end-effector position in world frame: p_ee(q)
    static Eigen::Vector3d forward_kinematics(const Eigen::Vector2d& q);

    // Plane constraint: g(q) = dot(normal, p_ee(q)) - offset
    static double evaluate_plane_constraint(const Eigen::Vector2d& q, const Eigen::Vector3d& normal, double offset);

    // Numerical Jacobian of plane constraint w.r.t q (1x2)
    static Eigen::Matrix<double, 1, 2> jacobian(const Eigen::Vector2d& q, const Eigen::Vector3d& normal, double offset);
};

// Independent Panda 7-DOF Reference Model
class Panda7Reference {
public:
    // Computes end-effector position in world frame: p_ee(q)
    static Eigen::Vector3d forward_kinematics(const Eigen::Matrix<double, 7, 1>& q);

    // Plane constraint: g(q) = dot(normal, p_ee(q)) - offset
    static double evaluate_plane_constraint(const Eigen::Matrix<double, 7, 1>& q, const Eigen::Vector3d& normal, double offset);

    // Numerical Jacobian of plane constraint w.r.t q (1x7)
    static Eigen::Matrix<double, 1, 7> jacobian(const Eigen::Matrix<double, 7, 1>& q, const Eigen::Vector3d& normal, double offset);
};

} // namespace cudro::reference
