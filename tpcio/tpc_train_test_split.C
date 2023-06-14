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

  Bool_t fSector = kTRUE;

  auto *inputFile = TFile::Open(inputfile, "READ");
  auto *tpcIni = inputFile->Get<TTree>("tpcIni");
  auto *tpcMov = inputFile->Get<TTree>("tpcMov");

  // select only same sector tracks (initrack)
  Long64_t counter;
  tpcMov->SetBranchAddress("counter",&counter);
  std::vector<short> *clSector = nullptr;
  tpcIni->SetBranchAddress("clSector",&clSector);


  for (Int_t i(0); i<tpcMov->GetEntries(); ++i){
    cout << "\rProcessing:" << i+1 << "/" << tpcMov->GetEntries() << flush;

    if (fSector){
      tpcMov->GetEntry(i);
      tpcIni->GetEntry(counter);

      
      if (std::adjacent_find(clSector->begin(), clSector->end(), std::not_equal_to<short>()) == clSector->end()){
        Float_t r = gRandom->Rndm();
          if (r<0.2){
            validList->Enter(i);
          }
          else{
            trainList->Enter(i);
          }
      }       
      else{
        continue;
      }
    
    }
    // end
  }
  cout << endl;

  cout << "Randomizing entries" << endl;

  // shuffle indices
  int ind[trainList->GetN()];
  std::iota(ind, ind + trainList->GetN(), 0);
  
  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(ind, ind + trainList->GetN(), g);

  auto *RtrainList = new TEntryList();

  for (int i : ind){
    cout << "\rShuffling index: " << i+1 << "/" << trainList->GetN() << flush;
    RtrainList->Enter(i);
  }
  cout << endl;

  cout << "Writing files" << endl;

  tpcMov->SetEntryList(RtrainList);
  auto trainFile = TFile::Open("train_sec_r.root", "RECREATE");
  // Because we set the list, only entry numbers in trainList are copied
  auto trainTree = tpcMov->CopyTree("");
  auto trainTreeIni = tpcIni->CopyTree("");
  trainFile->Write();
  trainFile->Close();


  tpcMov->SetEntryList(validList);
  auto validFile = TFile::Open("valid_sec_r.root", "RECREATE");
  // Because we set the list, only entry numbers in trainList are copied
  auto validTree = tpcMov->CopyTree("");
  auto validTreeIni = tpcIni->CopyTree("");
  validFile->Write();
  validFile->Close();

  cout << "Finished writing training and validation set" << endl;
}
