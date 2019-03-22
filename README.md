# VINS-MARGILIZATION




### vins边缘化代码整理
在vins中边缘化实质上就是构造了一个残差项类`MarginalizationFactor`加到系统整体的`costFuncction`中去，`MarginalizationFactor`类有一个`MarginalizationInfo`类指针，如下
``` cpp
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
```
`MarginalizationInfo*`指针用于最终的残差计算，本质上这两个类并没有太大的功能上的区别，只是用`MarginalizationInfo`方便对残差和参数快进行管理而已。`MarginalizationInfo`具体参数的含义：
而`MarginalizationInfo`是通过`ResidualBlockInfo`类进行保存要边缘化的残差项以及对雅克比的管理。下面是对这个类的详细说明：
```cpp
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;


	// 待优化的变量的数据存储，由ceres本身的二维数组形式变成vector<double*>
	// size是优化变量的数目的大小，存储的是每个待优化变量的地址。
    std::vector<double *> parameter_blocks;


	//　待marg的优化变量的id，边缘化就是在外部通过设置这个id选择marg的变量。
    std::vector<int> drop_set;

    double **raw_jacobians;


	// 之所以又重新定义了一个Eigen类型的jacobians，是由于raw_jacobians是指针数组，不方便在最后的Ｈ矩阵中参与计算，所以把他转换成Eigen的形式。
    //　可以看出jacobians.size() = 优化变量的个数。每个jacobians存储的是残差对这个变量的雅克比，
    // eg, 如果用三维参数点和七维的pose，残差是二维，那么jacobians[0]表示的是残差对pose的导数，是2*6维的矩阵，对应的raw_jacobians[0]中的十四维度的数组。
    // jacobians[1]表示的是残差对点的导数，是2*3维的矩阵，对应的raw_jacobians[1]中的六维度的数组。
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;


    // jacobians和residuals都会参与到最终的H矩阵构建。
    Eigen::VectorXd residuals;

    int localSize(int size){
		// 对于旋转来讲，实际上是se3的参数表达形式，所以有效的雅克比和参数维度只是６维
        return size == 7 ? 6 : size;
    }
};
```
在`ResidualBlockInfo`起着主要作用的函数就是`Evaluate()`，相当于把原始的costfunction中的参数块和雅克比转换成类里面变量的形式：
``` cpp
void ResidualBlockInfo::Evaluate()
{
	//　为残差容器分配空间
    residuals.resize(cost_function->num_residuals());

    // parameters has the same number of elements as parameter_block_sizes_.
    // 所以block_sizes通过ceres中costFunction自带函数返回实际的参数块相关的尺寸。
	// 第一个维度即block_sizes.size()表示优化参数的数量，
	// 第二个维度表示即block_sizes[i]表示这个优化变量的实际维度的大小。
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();

    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        //　这里调用的是eigen的resize 函数
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }　
	
	// 至此在ResidualBlockInfo类中就把所有需要在边缘化中进行保存的变量准备好了
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    if (loss_function)
    {　// loss部分省略　}
}
```
下面就是核心类`MarginalizationInfo`对整个边缘化进行管理和构建。


``` cpp
class MarginalizationInfo
{
  public:
    MarginalizationInfo()

    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();

    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    
    //m是要marg掉的变量的个数  是parameter_block_idx的总localsize  n是要保留的优化变量的个数
    int m, n;


	// 下面三个变量主要是用于管理优化变量的数据，主要作用是为了在构建H矩阵的时候保持一定的的顺序（marg变量在上分，保留下来的在下方） 它们的键都是优化变量的地址
	// parameter_block_size主要用于存储优化变量的维度
    std::unordered_map<long, int> parameter_block_size; //global size
	// parameter_block_idx可以理解为是parameter_block_size的一部分，主要是保存需要marg掉的变量和该变量的维度。
	// 注意parameter_block_idx一开始的存储的尺寸为0
    std::unordered_map<long, int> parameter_block_idx; //local size
	// 所有优化变量的数据容器
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

	 // H矩阵
    Eigen::MatrixXd linearized_jacobians;
    // b矩阵
    Eigen::VectorXd linearized_residuals;

};
```
与`ResidualBlockInfo`的主要接口是`std::vector<ResidualBlockInfo *> factors`这个容器，通过`addResidualBlockInfo`容器进行残差的添加，
``` cpp
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.push_back(residual_block_info);

    // book size and which ones to be marginalized
	// 第一步取出residual_block_info中管理的相关变量。
    // parameter_blocks存储优化变量数组的头指针，parameter_block_sizes存储每个优化变量的实际的维度。
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // record drop_set in parameter_block_idx
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
		// 注意这里初始化为0
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}
```
`preMarginalize()`函数很简单，主要是用于填充优化变量的数据容器`parameter_block_data`
``` cpp
void MarginalizationInfo::preMarginalize()
{
    // fill parameter_block_data according address
    for (std::vector<ResidualBlockInfo *>::iterator iter=factors.begin(); iter != factors.end(); ++iter)
    {
        ResidualBlockInfo * it = *iter;


		// 注意这里就决定了最终线性化的时刻的。
        it->Evaluate();

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                // copy the paramter
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}
```
核心函数是`marginalize()`函数，用于准备最终的H矩阵，
先构造H矩阵


``` cpp
void MarginalizationInfo::marginalize()
{
    // re-order the marginalization parts and residual parts
    // Marginalizations lie first, followed by the residual parts.
    // m for total size to marginalize; n for the totall size of residual
	// 第一步进行矩阵的重排。
	// 这样parameter_block_idx存储的尺寸就是要marg的变量的维度的叠加，这会在构造H矩阵的时候用到
    int pos = 0;
    for (std::map<long, int>::iterator it=parameter_block_idx.begin(); it != parameter_block_idx.end(); ++it)
    {
        it->second = pos;
        pos += localSize(parameter_block_size[it->first]);
    }

	// m = 要marg掉的变量的总维度
    m = pos; 

    // 这样parameter_block_idx就会拥有和parameter_block_size一样的尺寸，相当于进行扩容，而且保存的尺寸仍然是变量维度的叠加。
    for (std::map<long,int>::iterator it = parameter_block_size.begin(); it != parameter_block_size.end(); ++it)
    {
        if (parameter_block_idx.find(it->first) == parameter_block_idx.end())
        {
            // record residual parameter in parameter_block_idx
            parameter_block_idx[it->first] = pos;
            pos += localSize(it->second);
        }
    }
	// 至此parameter_block_idx保存的是所有优化变量的地址，键值是该变量在重排列好的矩阵的起始的维度。

    n = pos - m; // n是要保留的变量
    // pos is all m is marg  n is reserve

    // Form the Linear H * dx = -b0
	// 构建H矩阵
    Eigen::MatrixXd H(pos, pos);
    Eigen::VectorXd b(pos);
    H.setZero();
    b.setZero();
    for (std::vector<ResidualBlockInfo *>::iterator iter=factors.begin(); iter != factors.end(); ++iter)
    {   ResidualBlockInfo * it = *iter;
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
			// 注意这里idx_i
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
				// case:对角块
                if (i == j)
                    H.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {	// case: 非对角块
                    H.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    H.block(idx_j, idx_i, size_j, size_i) = H.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }


    Eigen::MatrixXd Hmm = 0.5 * (H.block(0, 0, m, m) + H.block(0, 0, m, m).transpose());
	// 计算伴随矩阵的特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Hmm);

	// 主要是为了求逆运算的稳定性
	// 这里用到了矩阵对角化的原理，恢复逆矩阵
    Eigen::MatrixXd Hmm_inv =
            saes.eigenvectors()
            * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
            * saes.eigenvectors().transpose();


	//    设x_{m}为要被marg掉的状态量，x_{r}是与x_{m}相关的状态量，所以在最后我们要保存的是x_{r}的信息
    //
    //      |      |    |          |   |
    //      |  Amm | Amr|  m       |bmm|        |x_{m}|
    //  A = |______|____|      b = |__ |       A|x_{r}| = b
    //      |  Arm | Arr|  n       |brr|
    //      |      |    |          |   |
    //  使用舒尔补:
    //  C = Arr - Arm*Amm^{-1}Amr
    //  d = brr - Arm*Amm^{-1}bmm
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Hmr = H.block(0, m, m, n); // equals is H.block<m , n>(0, m)
    Eigen::MatrixXd Hrm = H.block(m, 0, n, m);
    Eigen::MatrixXd Hrr = H.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    H = Hrr - Hrm * Hmm_inv * Hmr;
    b = brr - Hrm * Hmm_inv * bmm;

    // 求取A矩阵的特征值和特征值的倒数
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

	// 求取特征值的均方根
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // 利用舒尔补可得：
    // Ax_{r} = b ==>  UDU'x_{r}=b  ==> DU'x_{r}=U'b, 两边同乘√D^{-1},得
    //     √DU'x_{r} = √D^{-1}U'b
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;

}
```
至此H矩阵和b矩阵已经算好了，下一步是就是确定好哪些是要留下来的变量。
``` cpp
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();
    // 转存边缘化之后需要保留的参数块
    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {   // 根据矩阵重排里面的结果id进行挑选要保留下来的变量，根据上面的结果大于m的一定是需要保留下来的
			// it.first表示地址
            keep_block_size.push_back(parameter_block_size[it.first]); // 原始的尺寸大小
            keep_block_idx.push_back(parameter_block_idx[it.first]); // 累加后的参数块大小
            keep_block_data.push_back(parameter_block_data[it.first]); //地址
            keep_block_addr.push_back(addr_shift[it.first]); // 这里很重要：上面的三个容器相当于保存了所有的要保留的变量的相关信息，但是函数最终返回的是这个数组，在外面可以看出vins只传递了所有错过位置的pose
			// 所以可以理解为这个函数只返回不被marg的pose的地址
			// keep_block_data相当于存储的是带有先验信息的data
			// keep_block_data相当于存储的是为迭代第一次准备初始值的data。
        }
    }
    return keep_block_addr;
}
```
最终的边缘化求解
``` cpp
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
	

	// 第一步需要求解待更新变量的增量，增量永远是相对于线性化的那一个时刻进行求解的。
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
		// parameters永远存储的都是上一次最新迭代的变量结果
		// keep_block_data存储的是相当于带有先验的最原始的变量结果
		// 整个Evaluate函数就是为了求解b
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else // 对四元数进行处理
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
			// 这个positify函数没什么作用
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }	

	// 根据上面的推导，得出残差的形式。
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

	//雅克比永远是线性化那一时刻的雅克比
    if (jacobians)
    {
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }

    return true;
}


```

实际在边缘化过程中的代码：
``` cpp
 // 第一部分marg掉点
// 把投影相关的残差项直接加进去就好
// 设置要marg掉的变量是相机和点
// vins中做了一个这样的处理只添加起始帧是要被边缘化的旧帧且观测次数大于2的Features，即和别的帧有共视的点
ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function, vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]}, vector<int>{0, 3}); // 注意这里认为的是主导帧的pose和逆深度需要进行优化，投影帧的pose需要进行保留
marginalization_info->addResidualBlockInfo(residual_block_info);
```
// 第二部分保留之前的先验（这一部分也比较绕）
``` cpp
        // 先验误差会一直保存，而不是只使用一次
        //! 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)
if (last_marginalization_info)
{
	vector<int> drop_set;
    for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
    {
       //！查询待估计参数中是首帧状态量的序号	
		// 这么做的原因是因为在最后选择要保留的参数快的时候有一个移位的操作，所以还是需要判断第一帧进行边缘化
      if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
           last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
             drop_set.push_back(i);
       }

       //! 构造边缘化的的Factor
      // construct new marginlization_factor
       MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

       //! 添加上一次边缘化的参数块
      //！cost_function, loss_function, 待估计参数(last_marginalization_parameter_blocks, drop_set)
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

      marginalization_info->addResidualBlockInfo(residual_block_info);
}
```
最后添加要保留的参数块
``` cpp
        //! 这里是保存了所有状态量的信息，没有保存逆深度的状态量呢
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;

        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;


```
