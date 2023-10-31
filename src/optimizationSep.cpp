#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct X_Y_YAW_CostFunction
{
    X_Y_YAW_CostFunction(const pointTypeRGB &pointOri, const CoeffType &coeff)
        : pointOri_(pointOri), coeff_(coeff) {}

    template <typename T>
    bool operator()(const T *const transform, T *residuals) const
    {
        T srx = ceres::sin(transform[0]);
        T crx = ceres::cos(transform[0]);
        T sry = ceres::sin(transform[1]);
        T cry = ceres::cos(transform[1]);
        T srz = ceres::sin(transform[2]);
        T crz = ceres::cos(transform[2]);
        T tx = transform[3];
        T ty = transform[4];
        T tz = transform[5];

        T b1 = -crz * sry - cry * srx * srz;
        T b2 = cry * crz * srx - sry * srz;
        T b3 = crx * cry;
        T b4 = tx * -b1 + ty * -b2 + tz * b3;
        T b5 = cry * crz - srx * sry * srz;
        T b6 = cry * srz + crz * srx * sry;
        T b7 = crx * sry;
        T b8 = tz * b7 - ty * b6 - tx * b5;
        T c5 = crx * srz;

        T ary = (b1 * T(pointOri_.x) + b2 * T(pointOri_.y) - b3 * T(pointOri_.z) + b4) * T(coeff_.x) +
                (b5 * T(pointOri_.x) + b6 * T(pointOri_.y) - b7 * T(pointOri_.z) + b8) * T(coeff_.z);

        T atx = -b5 * T(coeff_.x) + c5 * T(coeff_.y) + b1 * T(coeff_.z);
        T atz = b7 * T(coeff_.x) - srx * T(coeff_.y) - b3 * T(coeff_.z);

        T d2 = T(coeff_.intensity);

        residuals[0] = ary;
        residuals[1] = atx;
        residuals[2] = atz;
        residuals[3] = -0.05 * d2;

        return true;
    }

private:
    const pointTypeRGB pointOri_;
    const CoeffType coeff_;
};

// Inside your optimization routine
ceres::Problem problem;

for (int i = 0; i < pointSelNum; i++)
{
    pointTypeRGB pointOri = laserCloudOri->points[i];
    CoeffType coeff = coeffSel->points[i];

    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<X_Y_YAW_CostFunction, 4, 6>(
        new X_Y_YAW_CostFunction(pointOri, coeff));

    problem.AddResidualBlock(cost_function, nullptr, transformCur);
}

ceres::Solver::Options options;
ceres::Solver::Summary summary;

ceres::Solve(options, &problem, &summary);

if (summary.termination_type == ceres::CONVERGENCE)
{
    return false;
}
else
{
    return true;
}
