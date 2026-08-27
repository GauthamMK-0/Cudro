#include <cudro/ast.hpp>

namespace cudro {

void JointDecl::accept(ASTVisitor& v) { v.visit(*this); }
void LinkDecl::accept(ASTVisitor& v) { v.visit(*this); }
void RobotDecl::accept(ASTVisitor& v) { v.visit(*this); }
void PlaneConstraint::accept(ASTVisitor& v) { v.visit(*this); }
void TaskDecl::accept(ASTVisitor& v) { v.visit(*this); }
void ClearanceDecl::accept(ASTVisitor& v) { v.visit(*this); }

void Spec::accept(ASTVisitor& v) { v.visit(*this); }

std::unique_ptr<JointDecl> make_joint(std::string name, std::string type,
                                       Vec3 axis, Vec3 origin,
                                       std::optional<std::pair<double,double>> limits, Location loc) {
    auto j = std::make_unique<JointDecl>();
    j->loc = loc;
    j->name = std::move(name);
    j->type = std::move(type);
    j->axis = axis;
    j->origin = origin;
    j->limits = limits;
    return j;
}

std::unique_ptr<LinkDecl> make_link(std::string name,
                                     std::vector<std::unique_ptr<Sphere>> spheres,
                                     std::optional<std::string> parent,
                                     std::optional<std::string> joint_ref, Location loc) {
    auto l = std::make_unique<LinkDecl>();
    l->loc = loc;
    l->name = std::move(name);
    l->spheres = std::move(spheres);
    l->parent = std::move(parent);
    l->joint_ref = std::move(joint_ref);
    return l;
}

std::unique_ptr<RobotDecl> make_robot(std::string name,
                                       std::vector<std::unique_ptr<JointDecl>> joints,
                                       std::vector<std::unique_ptr<LinkDecl>> links, Location loc) {
    auto r = std::make_unique<RobotDecl>();
    r->loc = loc;
    r->name = std::move(name);
    r->joints = std::move(joints);
    r->links = std::move(links);
    return r;
}

std::unique_ptr<PlaneConstraint> make_plane(std::string link, Vec3 point, Vec3 normal, double offset, Location loc) {
    auto p = std::make_unique<PlaneConstraint>();
    p->loc = loc;
    p->link = std::move(link);
    p->point_on_link = point;
    p->normal = normal;
    p->offset = offset;
    return p;
}

std::unique_ptr<TaskDecl> make_task(std::string name, std::string link,
                                     std::vector<std::unique_ptr<PlaneConstraint>> planes, Location loc) {
    auto t = std::make_unique<TaskDecl>();
    t->loc = loc;
    t->name = std::move(name);
    t->link = std::move(link);
    t->planes = std::move(planes);
    return t;
}

std::unique_ptr<ClearanceDecl> make_clearance(double min_distance, Location loc) {
    auto c = std::make_unique<ClearanceDecl>();
    c->loc = loc;
    c->min_distance = min_distance;
    return c;
}

} // namespace cudro