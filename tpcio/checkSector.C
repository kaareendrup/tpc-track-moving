#include <iostream>
#include <cassert>
#include <fstream>
#include <string>

using namespace std;

void checkSector(const char* inputfile){

    auto *inputFile = TFile::Open(inputfile, "READ");
    auto *tpcIni = inputFile->Get<TTree>("tpcIni");
    auto *tpcMov = inputFile->Get<TTree>("tpcMov");

    std::vector<short> *clSector = nullptr;
    tpcIni->SetBranchAddress("clSector",&clSector);

    Long64_t counter;
    tpcMov->SetBranchAddress("counter",&counter);

    for (Int_t i(0); i<tpcMov->GetEntries(); ++i){
    
        tpcMov->GetEntry(i);
        tpcIni->GetEntry(counter);

        if (std::adjacent_find(clSector->begin(), clSector->end(), std::not_equal_to<short>()) == clSector->end()){
          //all are the same
          continue;
        }
        else{
            cout << "Entry: " << counter << endl;
            cout << "Fix train test split!!" << endl;
            break;
        }
    }

}