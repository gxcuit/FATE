# Federated Machine Learning

[[中文](index.zh.md)]

FederatedML includes implementation of many common machine learning
algorithms on federated learning. All modules are developed in a
decoupling modular approach to enhance scalability. Specifically, we
provide:

1.  Federated Statistic: PSI, Union, Pearson Correlation, etc.
2.  Federated Feature Engineering: Feature Sampling, Feature Binning,
    Feature Selection, etc.
3.  Federated Machine Learning Algorithms: LR, GBDT, DNN,
    TransferLearning, which support Heterogeneous and Homogeneous
    styles.
4.  Model Evaluation: Binary | Multiclass | Regression | Clustering
    Evaluation, Local vs Federated Comparison.
5.  Secure Protocol: Provides multiple security protocols for secure
    multi-party computing and interaction between participants.

![Federated Machine Learning Framework](../images/federatedml_structure.png)

## Algorithm List

| Algorithm                                                                      | Module Name                  | Description                                                                                                                           | Data Input                                        | Data Output                                                                                           | Model Input                                          | Model Output                                                            |
| ------------------------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| Reader                                                                         | Reader                       | This component loads and transforms data from storage engine so that data is compatible with FATE computing engine                    | Original Data                                     | Transformed Data                                                                                      |                                                      |                                                                         |
| [DataIO](data_transform.md)                                                             | DataIO                       | This component transforms user-uploaded data into Instance object(deprecate in FATe-v1.7, use DataTransform instead).                 | Table, values are raw data.                       | Transformed Table, values are data instance defined [here](../python/federatedml/feature/instance.py) |                                                      | DataIO Model                                                            |
| [DataTransform](data_transform.md)                                                      | DataTransform                | This component transforms user-uploaded data into Instance object.                                                                    | Table, values are raw data.                       | Transformed Table, values are data instance defined [here](../python/federatedml/feature/instance.py) |                                                      | DataTransform Model                                                     |
| [Intersect](intersect.md)                                                     | Intersection                 | Compute intersect data set of multiple parties without leakage of difference set information. Mainly used in hetero scenario task.    | Table.                                            | Table with only common instance keys.                                                                 |                                                      | Intersect Model                                                         |
| [Federated Sampling](sample.md)                           | FederatedSample              | Federated Sampling data so that its distribution become balance in each party.This module supports standalone and federated versions. | Table                                             | Table of sampled data; both random and stratified sampling methods are supported.                     |                                                      |                                                                         |
| [Feature Scale](scale.md)                                     | FeatureScale                 | module for feature scaling and standardization.                                                                                       | Table，values are instances.                       | Transformed Table.                                                                                    | Transform factors like min/max, mean/std.            |                                                                         |
| [Hetero Feature Binning](feature_binning.md)                   | Hetero Feature Binning       | With binning input data, calculates each column's iv and woe and transform data according to the binned information.                  | Table, values are instances.                      | Transformed Table.                                                                                    |                                                      | iv/woe, split points, event count, non-event count etc. of each column. |
| [Homo Feature Binning](feature_binning.md)                                            | Homo Feature Binning         | Calculate quantile binning through multiple parties                                                                                   | Table                                             | Transformed Table                                                                                     |                                                      | Split points of each column                                             |
| [OneHot Encoder](onehot_encoder.md)                                   | OneHotEncoder                | Transfer a column into one-hot format.                                                                                                | Table, values are instances.                      | Transformed Table with new header.                                                                    |                                                      | Feature-name mapping between original header and new header.            |
| [Hetero Feature Selection](feature_selection.md)               | HeteroFeatureSelection       | Provide 5 types of filters. Each filters can select columns according to user config                                                  | Table                                             | Transformed Table with new header and filtered data instance.                                         | If iv filters used, hetero\_binning model is needed. | Whether each column is filtered.                                        |
| [Union](union.md)                                                             | Union                        | Combine multiple data tables into one.                                                                                                | Tables.                                           | Table with combined values from input Tables.                                                         |                                                      |                                                                         |
| [Hetero-LR](logistic_regression.md)                                          | HeteroLR                     | Build hetero logistic regression model through multiple parties.                                                                      | Table, values are instances                       | Table, values are instances.                                                                          |                                                      | Logistic Regression Model, consists of model-meta and model-param.      |
| [Local Baseline](local_baseline.md)                                           | LocalBaseline                | Wrapper that runs sklearn(scikit-learn) Logistic Regression model with local data.                                                    | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      |                                                                         |
| [Hetero-LinR](linear_regression.md)                                           | HeteroLinR                   | Build hetero linear regression model through multiple parties.                                                                        | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | Linear Regression Model, consists of model-meta and model-param.        |
| [Hetero-Poisson](poisson_regression.md)                                       | HeteroPoisson                | Build hetero poisson regression model through multiple parties.                                                                       | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | Poisson Regression Model, consists of model-meta and model-param.       |
| [Homo-LR](logistic_regression.md)                                             | HomoLR                       | Build homo logistic regression model through multiple parties.                                                                        | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | Logistic Regression Model, consists of model-meta and model-param.      |
| [Homo-NN](homo_nn.md)                                                         | HomoNN                       | Build homo neural network model through multiple parties.                                                                             | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | Neural Network Model, consists of model-meta and model-param.           |
| [Hetero Secure Boosting](ensemble.md)                                         | HeteroSecureBoost            | Build hetero secure boosting model through multiple parties                                                                           | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | SecureBoost Model, consists of model-meta and model-param.              |
| [Hetero Fast Secure Boosting](ensemble.md)                                    | HeteroFastSecureBoost        | Build hetero secure boosting model through multiple parties in layered/mix manners.                                                   | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | FastSecureBoost Model, consists of model-meta and model-param.          |
| [Hetero Secure Boost Feature Transformer](sbt_feature_transformer.md) | SBT Feature Transformer      | This component can encode sample using Hetero SBT leaf indices.                                                                       | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | SBT Transformer Model                                                   |
| [Evaluation](evaluation.md)                                                   | Evaluation                   | Output the model evaluation metrics for user.                                                                                         | Table(s), values are instances.                   |                                                                                                       |                                                      |                                                                         |
| [Hetero Pearson](correlation.md)                                              | HeteroPearson                | Calculate hetero correlation of features from different parties.                                                                      | Table, values are instances.                      |                                                                                                       |                                                      |                                                                         |
| [Hetero-NN](hetero_nn.md)                                                     | HeteroNN                     | Build hetero neural network model.                                                                                                    | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | Hetero Neural Network Model, consists of model-meta and model-param.    |
| [Homo Secure Boosting](ensemble.md)                                           | HomoSecureBoost              | Build homo secure boosting model through multiple parties                                                                             | Table, values are instances.                      | Table, values are instances.                                                                          |                                                      | SecureBoost Model, consists of model-meta and model-param.              |
| [Homo OneHot Encoder](homo_onehot_encoder.md)                         | HomoOneHotEncoder            | Build homo onehot encoder model through multiple parties.                                                                             | Table, values are instances.                      | Transformed Table with new header.                                                                    |                                                      | Feature-name mapping between original header and new header.            |
| [Data Split](data_split.md)                                                   | Data Split                   | Split one data table into 3 tables by given ratio or count                                                                            | Table, values are instances.                      | 3 Tables, values are instance.                                                                        |                                                      |                                                                         |
| [Column Expand](column-expand.md)                                     | Column Expand                | Add arbitrary number of columns with user-provided values.                                                                            | Table, values are raw data.                       | Transformed Table with added column(s) and new header.                                                |                                                      | Column Expand Model                                                     |
| [Secure Information Retrieval](sir.md)                                        | Secure Information Retrieval | Securely retrieves information from host through oblivious transfer                                                                   | Table, values are instance                        | Table, values are instance                                                                            |                                                      |                                                                         |
| [Hetero Federated Transfer Learning](hetero_ftl.md)                           | Hetero FTL                   | Build Hetero FTL Model Between 2 party                                                                                                | Table, values are instance                        |                                                                                                       |                                                      | Hetero FTL Model                                                        |
| [Hetero KMeans](hetero_kmeans.md)                                             | Hetero KMeans                | Build Hetero KMeans model through multiple parties                                                                                    | Table, values are instance                        | Table, values are instance; Arbier outputs 2 Tables                                                   |                                                      | Hetero KMeans Model                                                     |
| [PSI](psi.md)                                                                 | PSI module                   | Compute PSI value of features between two table                                                                                       | Table, values are instance                        |                                                                                                       |                                                      | PSI Results                                                             |
| [Data Statistics](statistic.md)                                               | Data Statistics              | This component will do some statistical work on the data, including statistical mean, maximum and minimum, median, etc.               | Table, values are instance                        | Table                                                                                                 |                                                      | Statistic Result                                                        |
| [Scorecard](scorecard.md)                                                     | Scorecard                    | Scale predict score to credit score by given scaling parameters                                                                       | Table, values are predict score                   | Table, values are score results                                                                       |                                                      |                                                                         |
| [Sample Weight](sample_weight.md)                                        | Sample Weight                | Assign weight to instances according to user-specified parameters                                                                     | Table, values are instance                        | Table, values are weighted instance                                                                   |                                                      | SampleWeight Model                                                      |
| [Feldman Verifiable Sum](feldman_verifiable_sum.md)                           | Feldman Verifiable Sum       | This component will sum multiple privacy values without exposing data                                                                 | Table, values to sum                              | Table, values are sum results                                                                         |                                                      |                                                                         |
| [Feature Imputation](feature_imputation.md)                           | Feature Imputation           | This component imputes missing features using arbitrary methods/values                                                                | Table, values are Instances                       | Table, values with missing features filled                                                            |                                                      | FeatureImputation Model                                                 |
| [Label Transform](label_transform.md)                                    | Label Transform              | Replaces label values of input data instances and predict results                                                                     | Table, values are Instances or prediction results | Table, values with transformed label values                                                           |                                                      | LabelTransform Model                                                    |
| [Hetero SSHE Logistic Regression](sshe_lr.md)                                    | Hetero SSHE LR            | Build hetero logistic regression model without arbiter                                                                     | Table, values are Instances                               | Table, values are Instances                                                           |                                                      | SSHE LR Model                                                    |

## Secure Protocol

  - [Encrypt](secureprotol.md#encrypt)
      - [Paillier encryption](secureprotol.md#paillier-encryption)
      - [Affine Homomorphic Encryption](secureprotol.md#affine-homomorphic-encryption)
      - [IterativeAffine Homomorphic Encryption](secureprotol.md#iterativeaffine-homomorphic-encryption)
      - [RSA encryption](secureprotol.md#rst-encryption)
      - [Fake encryption](secureprotol.md#fake-encryption)
  - [Encode](secureprotol.md#encode)
  - [Diffne Hellman Key Exchange](secureprotol.md#diffne-hellman-key-exchange)
  - [SecretShare MPC Protocol(SPDZ)](secureprotol.md#secretshare-mpc-protocol-spdz)
  - [Oblivious Transfer](secureprotol.md#oblivious-transfer)
  - [Feldman Verifiable Secret Sharing](secureprotol.md#feldman-verifiable-secret-sharing)

## Params

::: federatedml.param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false