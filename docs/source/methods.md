# Implemented methods
The following methods are currently implemented in `probly`.

## Representation
### Second-order Distributions
These methods represent (epistemic) uncertainty by a second-order distribution over distributions.
#### Bayesian Neural Networks
{cite:p}`blundellWeightUncertainty2015`

#### Dropout
{cite:p}`galDropoutBayesian2016`

#### DropConnect
{cite:p}`mobinyDropConnectEffective2019`

#### Deep Ensembles
{cite:p}`lakshminarayananSimpleScalable2017`

#### Evidential Deep Learning
{cite:p}`sensoyEvidentialDeep2018`
{cite:p}`aminiDeepEvidential2020`

### Credal Sets
These methods represent (epistemic) uncertainty by a convex set of distributions.
#### Credal Ensembling
{cite:p}`nguyenCredalEnsembling2025`

## Quantification
#### Upper / lower entropy
{cite:p}`abellanDisaggregatedTotal2006`

#### Generalized Hartley
{cite:p}`abellanNonSpecificity2000`

#### Entropy-based
{cite:p}`depewegDecompositionUncertainty2018`

#### Distance-based
{cite:p}`saleSecondOrder2024`

### Conformal Prediction
These methods represent uncertainty by a set of predictions.
#### Split Conformal Prediction
{cite:p}`angelopoulosGentleIntroduction2021`
## Calibration
These methods adjust the model's probabilities to better reflect the true probabilities.
#### Focal Loss
{cite:p}`linFocalLoss2017`

#### Label Relaxation
{cite:p}`lienenFromLabel2021`

#### Temperature Scaling
{cite:p}`guoOnCalibration2017`
