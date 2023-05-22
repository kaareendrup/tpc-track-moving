#include <iostream>
#include <cassert>
#include <fstream>
#include <string>

using namespace std;

// template<typename  T>
void tpc_train_test_split(const char* inputfile)//, const char* savepath)
{
  auto *trainList = new TEntryList();
  auto *validList = new TEntryList();



  auto *inputFile = TFile::Open(inputfile, "READ");
  auto *tpcIni = inputFile->Get<TTree>("tpcIni");
  auto *tpcMov = inputFile->Get<TTree>("tpcMov");

  for (Int_t i(0); i<tpcMov->GetEntries(); ++i){
    cout << "\r" << i << "/" << tpcMov->GetEntries() << flush;
    Float_t r = gRandom->Rndm();
    if (r<0.2){
      validList->Enter(i);
    }
    else{
      trainList->Enter(i);
    }

  }
  cout << endl;

  tpcMov->SetEntryList(trainList);
  auto trainFile = TFile::Open("train.root", "RECREATE");
  // Because we set the list, only entry numbers in trainList are copied
  auto trainTree = tpcMov->CopyTree("");
  auto trainTreeIni = tpcIni->CopyTree("");
  trainFile->Write();
  trainFile->Close();


  tpcMov->SetEntryList(validList);
  auto validFile = TFile::Open("valid.root", "RECREATE");
  // Because we set the list, only entry numbers in trainList are copied
  auto validTree = tpcMov->CopyTree("");
  auto validTreeIni = tpcIni->CopyTree("");
  validFile->Write();
  validFile->Close();
}
