#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

int main()
{
    cout << "Time taken by function: " << endl;

    int sz = 10000000;
    int *randArray = new int[sz];
    for (int i = 0; i < sz; i++)
        randArray[i] = rand() % 100; //Generate number between 0 to 99

    auto start = high_resolution_clock::now();

    for (int i = 0; i < sz; i++)
    {
        bool v = randArray[i] > 50;
    }
    auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start);

    cout << "Time taken by function: "
         << duration.count()
         << " microseconds" << endl;

    return 0;
}