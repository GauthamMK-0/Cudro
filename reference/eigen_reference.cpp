#include "eigen_reference.hpp"

#include <cmath>

namespace cudro::reference {

static Eigen::Matrix4d rot_x(double angle) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    double c = std::cos(angle), s = std::sin(angle);
    m(1, 1) = c;  m(1, 2) = -s;
    m(2, 1) = s;  m(2, 2) = c;
    return m;
}

static Eigen::Matrix4d rot_y(double angle) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    double c = std::cos(angle), s = std::sin(angle);
    m(0, 0) = c;  m(0, 2) = s;
    m(2, 0) = -s; m(2, 2) = c;
    return m;
}

static Eigen::Matrix4d rot_z(double angle) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    double c = std::cos(angle), s = std::sin(angle);
    m(0, 0) = c;  m(0, 1) = -s;
    m(1, 0) = s;  m(1, 1) = c;
    return m;
}

static Eigen::Matrix4d translate(double tx, double ty, double tz) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m(0, 3) = tx;
    m(1, 3) = ty;
    m(2, 3) = tz;
    return m;
}

// Planar 2R Reference Implementation
Eigen::Vector3d Planar2RReference::forward_kinematics(const Eigen::Vector2d& q) {
    Eigen::Matrix4d Tj1 = rot_z(q(0)) * translate(0.0, 0.0, 0.0);
    Eigen::Matrix4d Tj2 = rot_z(q(1)) * translate(1.0, 0.0, 0.0);

    Eigen::Matrix4d T_base = Tj1;
    Eigen::Matrix4d T_arm1 = T_base * Tj2;
    Eigen::Matrix4d T_ee = T_arm1;

    Eigen::Vector4d p_local(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d p_world = T_ee * p_local;
    return p_world.head<3>();
}

double Planar2RReference::evaluate_plane_constraint(const Eigen::Vector2d& q, const Eigen::Vector3d& normal, double offset) {
    Eigen::Vector3d p_world = forward_kinematics(q);
    return normal.dot(p_world) - offset;
}

Eigen::Matrix<double, 1, 2> Planar2RReference::jacobian(const Eigen::Vector2d& q, const Eigen::Vector3d& normal, double offset) {
    Eigen::Matrix<double, 1, 2> J;
    const double h = 1e-6;
    double g0 = evaluate_plane_constraint(q, normal, offset);
    for (int i = 0; i < 2; ++i) {
        Eigen::Vector2d q_pert = q;
        q_pert(i) += h;
        double g_pert = evaluate_plane_constraint(q_pert, normal, offset);
        J(0, i) = (g_pert - g0) / h;
    }
    return J;
}

// Panda 7-DOF Reference Implementation
Eigen::Vector3d Panda7Reference::forward_kinematics(const Eigen::Matrix<double, 7, 1>& q) {
    Eigen::Matrix4d Tj1 = rot_z(q(0)) * translate(0.0, 0.0, 0.333);
    Eigen::Matrix4d Tj2 = rot_y(q(1)) * translate(0.0, 0.0, 0.0);
    Eigen::Matrix4d Tj3 = rot_z(q(2)) * translate(0.0, -0.316, 0.0);
    Eigen::Matrix4d Tj5 = rot_z(q(4)) * translate(0.0, 0.384, 0.0);
    Eigen::Matrix4d Tj7 = rot_z(q(6)) * translate(0.0, 0.107, 0.0);

    Eigen::Matrix4d T_base = Tj1;
    Eigen::Matrix4d T_arm1 = T_base * Tj2;
    Eigen::Matrix4d T_arm2 = T_arm1 * Tj3;
    Eigen::Matrix4d T_forearm = T_arm2 * Tj5;
    Eigen::Matrix4d T_ee = T_forearm * Tj7;

    Eigen::Vector4d p_local(0.0, 0.0, 0.02, 1.0);
    Eigen::Vector4d p_world = T_ee * p_local;
    return p_world.head<3>();
}

double Panda7Reference::evaluate_plane_constraint(const Eigen::Matrix<double, 7, 1>& q, const Eigen::Vector3d& normal, double offset) {
    Eigen::Vector3d p_world = forward_kinematics(q);
    return normal.dot(p_world) - offset;
}

Eigen::Matrix<double, 1, 7> Panda7Reference::jacobian(const Eigen::Matrix<double, 7, 1>& q, const Eigen::Vector3d& normal, double offset) {
    Eigen::Matrix<double, 1, 7> J;
    const double h = 1e-6;
    double g0 = evaluate_plane_constraint(q, normal, offset);
    for (int i = 0; i < 7; ++i) {
        Eigen::Matrix<double, 7, 1> q_pert = q;
        q_pert(i) += h;
        double g_pert = evaluate_plane_constraint(q_pert, normal, offset);
        J(0, i) = (g_pert - g0) / h;
    }
    return J;
}

} // namespace cudro::reference
