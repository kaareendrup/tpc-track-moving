#include <iostream>
#include <cassert>
#include <fstream>
#include <string>

#include <algorithm>
#include <random>

using namespace std;

// template<typename  T>
void tpc_train_test_split(const char* inputfile)//, const char* savepath)
{
  const char* fileName = "n6_18082023_iniRef_Z_Above0_Tgl_Above0_dz_shift_100_150";;
  // const char* fileName = "n6_18082023_iniRef_Z_Above0_Tgl_Above0_dz_positiveshift";
  // const char* fileName = "n6_18082023_iniRef_Z_Above0_Tgl_Above0_dz_negativeshift";


  auto *trainList = new TEntryList();
  auto *validList = new TEntryList();

  auto* trainListIni = new TEntryList();
  auto* validListIni = new TEntryList();


  auto *inputFile = TFile::Open(inputfile, "READ");
  auto *tpcIni = inputFile->Get<TTree>("tpcIni");
  auto *tpcMov = inputFile->Get<TTree>("tpcMov");

  o2::track::TrackParCov *iniTrackRef = nullptr;
  float bcTB, dz,T0;
  tpcIni->SetBranchAddress("iniTrackRef",&iniTrackRef);

  tpcMov->SetBranchAddress("dz",&dz);
  // select only same sector tracks (initrack)
  Long64_t counter;
  Long64_t ini_counter;
  tpcMov->SetBranchAddress("counter",&counter);
  tpcIni->SetBranchAddress("counter",&ini_counter);
  std::vector<short> *clSector = nullptr;
  tpcIni->SetBranchAddress("clSector",&clSector);



  for (Int_t i(0); i<tpcMov->GetEntries(); ++i){
    cout << "\rProcessing:" << i+1 << "/" << tpcMov->GetEntries() << flush;

    tpcMov->GetEntry(i);
    if (dz < 100 || dz > 150) continue;
    tpcIni->GetEntry(counter);
    // track restrictions
    if (iniTrackRef->getZ()<0) continue;
    if (iniTrackRef->getTgl()<0) continue;
    
    
    // if (std::adjacent_find(clSector->begin(), clSector->end(), std::not_equal_to<short>()) == clSector->end()){
    Float_t r = gRandom->Rndm();
    if (r<0.2){
      validList->Enter(i);
      validListIni->Enter(counter);
    }
    else{
      trainList->Enter(i);
      trainListIni->Enter(counter);
    }

    
  }
  cout << endl;

  // for (int i(0); i < trainList->GetN();++i){
  //   cout << trainList->GetEntry(i) << endl;
  // }


  cout << "Randomizing entries" << endl;

  cout << "TrainList N: " << trainList->GetN() << endl;

  // shuffle indices
  // int ind[trainList->GetN()];
  // std::iota(ind, ind + trainList->GetN(), 0); // evenly spaces values - men skal jo bruge trainlist..
  std::vector<int> ind;
  for (int i(0); i < trainList->GetN(); ++i){
    ind.push_back(trainList->GetEntry(i));
  }

  
  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(ind.begin(), ind.end(), g);

  auto *RtrainList = new TEntryList();

  for (int i : ind){
    cout << "\rShuffling index: " << i+1 << "/" << trainList->GetN() << flush;
    RtrainList->Enter(i);
  }
  cout << endl;


  cout << "Writing files" << endl;
  tpcMov->SetEntryList(RtrainList);
  // tpcIni->SetEntryList(trainListIni);
  auto trainFile = TFile::Open(Form("train_%s.root",fileName), "RECREATE");
  //// Because we set the list, only entry numbers in trainList are copied
  auto trainTree = tpcMov->CopyTree("");
  auto trainTreeIni = tpcIni->CopyTree("");
  trainFile->Write(0,TObject::kOverwrite);
  trainFile->Close();


  tpcMov->SetEntryList(validList);
  // tpcIni->SetEntryList(validListIni);
  auto validFile = TFile::Open(Form("valid_%s.root",fileName), "RECREATE");
  // Because we set the list, only entry numbers in trainList are copied
  auto validTree = tpcMov->CopyTree("");
  auto validTreeIni = tpcIni->CopyTree("");
  validFile->Write(0,TObject::kOverwrite);
  validFile->Close();

  cout << "Finished writing training and validation set" << endl;
}
