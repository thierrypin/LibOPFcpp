
#include <iostream>
#include <fstream>
#include <vector>

#include <libopfcpp/OPF.hpp>
#include <libopfcpp/util.hpp>

using namespace std;
using namespace opf;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        string usage = string("Usage: ") + string(argv[0]) + " input1.dat [input2.dat [...]]";
        cout << usage << endl;
        return -1;
    }

    for (int f = 1; f < argc; f++)
    {
        string input_path = argv[f];
        string output_path = input_path.substr(0, input_path.find_last_of('.')) + ".csv";
        cout << input_path << " -> " << output_path << endl;

        Mat<float> data;
        vector<int> labels;
        read_mat_labels(input_path, data, labels);

        std::ofstream file(output_path, std::ios::out);
        for (int i = 0; i < data.rows; i++)
        {
            float* row = data.row(i);
            file << labels[i];
            for (int j = 0; j < data.cols; j++)
            {
                file << ";" << row[j];
            }
            file << "\n";
        }
        file.close();
    }

    return 0;
}

