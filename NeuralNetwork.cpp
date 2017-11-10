#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>


struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned index);
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    void setOutputVal(double val) { outputVal = val; }
    double getOutputVal(void) const { return outputVal; }

private:
    static double eta;
    static double alpha;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);

    double sumDOW(const Layer &nextLayer) const;

    double outputVal;
    std::vector<Connection> outputWeights;
    unsigned myIndex;
    double gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    for(unsigned n = 0; n < nextLayer.size(); ++n) {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }

    return sum;
}

Neuron::Neuron(unsigned numOutputs, unsigned index)
{
    for(unsigned c = 0; c < numOutputs; ++c) {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }

    myIndex = index;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].outputWeights[myIndex].weight;
    }

    outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - outputVal;
    gradient = delta * Neuron::transferFunctionDerivative(outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

        double newDeltaWeight =
            eta
            * neuron.getOutputVal()
            * gradient
            * alpha
            * oldDeltaWeight;

        neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
        neuron.outputWeights[myIndex].weight += newDeltaWeight;
    }
}


class Net {
public:
    Net(const std::vector<unsigned> topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;

private:
    std::vector<Layer> layers;
    double error;

    double recentAverageError;
    double recentAverageSmoothingFactor;
};

Net::Net(const std::vector<unsigned> topology)
{
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a Neuron!" << std::endl;
        }

        layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == layers[0].size() - 1);

    for(unsigned i = 0; i < inputVals.size(); ++i) {
        layers[0][i].setOutputVal(inputVals[i]);
    }

    for(unsigned layerNum = 0; layerNum < layers.size(); ++layerNum) {
        Layer &prevLayer = layers[layerNum -1];
        for(unsigned n = 0; n < layers[layerNum].size(); ++n) {
            layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    Layer &outputLayer = layers.back();
    error = 0.0;

    for(unsigned n = 0; n < outputLayer.size(); ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error += delta * delta;
    }

    error /= outputLayer.size() - 1;
    error = sqrt(error);

    recentAverageError =
        (recentAverageError * recentAverageSmoothingFactor + error)
        / (recentAverageSmoothingFactor + 1.0);

    for(unsigned n = 0; n < outputLayer.size(); ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for(unsigned layerNum = 0; layerNum < layers.size() - 2; ++layerNum) {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for(unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];

        for(unsigned n = 0; n < layers.size(); ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const
{
    resultVals.clear();

    for(unsigned n = 0; n < layers.back().size() - 1; ++n) {
        resultVals.push_back(layers.back()[n].getOutputVal());
    }
}


int main(int argc, char const *argv[])
{
    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);

    std::vector<double> inputVals;
    myNet.feedForward(inputVals);
    std::vector<double> targetVals;
    myNet.backProp(targetVals);
    std::vector<double> resultVals;
    myNet.getResults(resultVals);

    return 0;
}
