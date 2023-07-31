#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

static std::vector<std::vector<Eigen::Triplet<double, int>>> get_normalized_laplacian(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector< std::vector<int> >& edges)
{
    unsigned nv = unsigned(vertices.size());
    std::vector<std::vector<Eigen::Triplet<double, int>>> mat_elemts(nv);
    for (int i = 0; i < nv; ++i)
        mat_elemts[i].reserve(10);

    for (int i = 0; i < nv; ++i)
    {
        const Eigen::Vector3d c_pos = vertices[i];

        //get laplacian
        double sum = 0.;
        int nb_edges = edges[i].size();
        for (int e = 0; e < nb_edges; ++e)
        {
            int next_edge = (e + 1) % nb_edges;
            int prev_edge = (e + nb_edges - 1) % nb_edges;

            Eigen::Vector3d v1 = c_pos - vertices[edges[i][prev_edge]];
            Eigen::Vector3d v2 = vertices[edges[i][e]] - vertices[edges[i][prev_edge]];
            Eigen::Vector3d v3 = c_pos - vertices[edges[i][next_edge]];
            Eigen::Vector3d v4 = vertices[edges[i][e]] - vertices[edges[i][next_edge]];

            double cotan1 = (v1.dot(v2)) / (1e-6 + (v1.cross(v2)).norm());
            double cotan2 = (v3.dot(v4)) / (1e-6 + (v3.cross(v4)).norm());

            double w = (cotan1 + cotan2) * 0.5;
            sum += w;
            mat_elemts[i].push_back(Eigen::Triplet<double, int>(i, edges[i][e], w));
        }

        for (Eigen::Triplet<double, int>& t : mat_elemts[i])
            t = Eigen::Triplet<double, int>(t.row(), t.col(), t.value() / sum);

        mat_elemts[i].push_back(Eigen::Triplet<double, int>(i, i, -1.0));
    }
    return mat_elemts;
}

std::vector<Eigen::Vector3d> implicit_laplacian_smoothing(
    const std::vector<Eigen::Vector3d>& in_vertices,
    const std::vector< std::vector<int> >& edges,
    int nb_iter,
    float alpha)
{
    unsigned nb_vertices = unsigned(in_vertices.size());

    std::vector<Eigen::VectorXd> xyz;
    std::vector<Eigen::VectorXd> rhs;

    xyz.resize(3, Eigen::VectorXd::Zero(nb_vertices));
    rhs.resize(3, Eigen::VectorXd::Zero(nb_vertices));

    for (int i = 0; i < nb_vertices; ++i)
    {
        Eigen::Vector3d pos = in_vertices[i];
        rhs[0][i] = pos.x();
        rhs[1][i] = pos.y();
        rhs[2][i] = pos.z();
    }


    // Build laplacian
    std::vector<std::vector<Eigen::Triplet<double, int>>> mat_elemts = get_normalized_laplacian(in_vertices, edges);
    Eigen::SparseMatrix<double> L(nb_vertices, nb_vertices);
    std::vector<Eigen::Triplet<double, int>> triplets;
    triplets.reserve(nb_vertices * 10);
    for (const std::vector<Eigen::Triplet<double, int>>& row : mat_elemts)
        for (const Eigen::Triplet<double, int>& elt : row)
            triplets.push_back(elt);

    L.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<double> I = Eigen::MatrixXd::Identity(nb_vertices, nb_vertices).sparseView();
    L = I - L * alpha;

    L = L * L * L;

    // Solve for x, y, z
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    std::cout << "solver.info() == Eigen::Success: " << (solver.info() == Eigen::Success) << std::endl;

    for (int k = 0; k < 3; k++) {
        xyz[k] = solver.solve(rhs[k]);
    }

    std::vector<Eigen::Vector3d> out_verts(nb_vertices);
    for (int i = 0; i < nb_vertices; ++i) {
        Eigen::Vector3d v(xyz[0][i], xyz[1][i], xyz[2][i]);
        out_verts[i] = v;
    }

    return out_verts;
}