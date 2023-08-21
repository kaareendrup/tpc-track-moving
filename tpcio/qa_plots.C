#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include "TMath.h"

using namespace std;

TCanvas scatter(TTree* tpc, const char* cc, const char* track1, const char* track2)
{
    auto *c1 = new TCanvas(Form("%s",cc),Form("%s",cc),800,600);
    c1->Divide(3,2);
    c1->cd(1);
    tpc->Draw(Form("%s->getZ() : %s->getZ()",track1,track2));
    c1->cd(2);
    tpc->Draw(Form("%s->getY() : %s->getY()",track1,track2));
    c1->cd(3);
    tpc->Draw(Form("%s->getSnp() : %s->getSnp()",track1,track2));
    c1->cd(4);
    tpc->Draw(Form("%s->getTgl() : %s->getTgl()",track1,track2));
    c1->cd(5);
    tpc->Draw(Form("%s->getQ2Pt() : %s->getQ2Pt()",track1,track2));
    c1->cd(6);
    tpc->Draw(Form("%s->getZ() - dz: %s->getZ()",track1,track2));

    return c1;
}

TString RMSE(const char* t1, const char* t2, const char* param)
{
    TString stringo = Form("TMath::Sqrt(TMath::Power(%s->get%s - %s->get%s,2)) : %s->get%s",t1,param,t2,param,t1,param);
    return stringo;
}

TCanvas sigma(TTree *tpc, const char* cc, const char* track1, const char* track2)
{
    gStyle->SetPalette(1);
    auto *c1 = new TCanvas(Form("%s",cc),Form("%s",cc),800,600);
    c1->Divide(3,2);
    c1->cd(1);
    tpc->Draw(RMSE(track1,track2,"Y()"),"","coly");
    c1->cd(2);
    tpc->Draw(RMSE(track1,track2,"Z()"),"","coly");
    c1->cd(3);
    tpc->Draw(RMSE(track1,track2,"Snp()"),"","coly");
    c1->cd(4);
    tpc->Draw(RMSE(track1,track2,"Tgl()"),"","coly");
    c1->cd(5);
    tpc->Draw(RMSE(track1,track2,"Q2Pt()"),"","coly");

    return c1;
}

TCanvas difference1D(TTree* tpc, const char* cc, const char* tr1, const char* tr2)
{
    gStyle->SetPalette(1);
    auto *c1 = new TCanvas(Form("%s",cc),Form("%s",cc),800,600);
    c1->Divide(3,2);
    c1->cd(1);
    gPad->SetLogy(1);
    tpc->Draw(Form("%s->getY() - %s->getY()",tr1,tr2));
    c1->cd(2);
    gPad->SetLogy(1);
    tpc->Draw(Form("%s->getZ() - %s->getZ()",tr1,tr2));
    c1->cd(3);
    gPad->SetLogy(1);
    tpc->Draw(Form("%s->getSnp() - %s->getSnp()",tr1,tr2));
    c1->cd(4);
    gPad->SetLogy(1);
    tpc->Draw(Form("%s->getTgl() - %s->getTgl()",tr1,tr2));
    c1->cd(5);
    gPad->SetLogy(1);
    tpc->Draw(Form("%s->getQ2Pt() - %s->getQ2Pt()",tr1,tr2));

    return c1;
}

TCanvas iniMov(TTree* tpc, const char* cc, const char* tr1, const char* tr2)
{
    gStyle->SetPalette(1);
    auto *c1 = new TCanvas(Form("%s",cc),Form("%s",cc),800,600);
    c1->Divide(3,2);
    c1->cd(1);
    tpc->Draw(Form("%s->getY() - %s->getY() : %s->getY()",tr1,tr2,tr1));
    c1->cd(2);
    tpc->Draw(Form("%s->getZ() - %s->getZ() - dz : %s->getZ()",tr1,tr2,tr1));
    c1->cd(3);
    tpc->Draw(Form("%s->getSnp() - %s->getSnp() : %s->getSnp()",tr1,tr2,tr1));
    c1->cd(4);
    tpc->Draw(Form("%s->getTgl() - %s->getTgl() : %s->getTgl()",tr1,tr2,tr1));
    c1->cd(5);
    tpc->Draw(Form("%s->getQ2Pt() - %s->getQ2Pt() : %s->getQ2Pt()",tr1,tr2,tr1));

    return c1;
}

// void qa_plots(const char* inputfile)//, const char* savepath)
void qa_plots()
{
    
    auto *inputFile = TFile::Open("/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/QA_tpctrack/tpc-trackStudy_15082023_nm2_n0.root", "READ");
    auto *tpc = inputFile->Get<TTree>("tpc");

    // o2::tpc::TrackTPC* iniTrack = nullptr;
    // o2::track::TrackParCov *iniTrackRef = nullptr, * movTrackRef = nullptr, *mcTrack = nullptr;
    // float bcTB, dz,T0;

    // tpc->SetBranchAddress("iniTrack",&iniTrack);
    // tpc->SetBranchAddress("iniTrackRef",&iniTrackRef);
    // tpc->SetBranchAddress("movTrackRef",&movTrackRef);
    // tpc->SetBranchAddress("dz",&dz);
    // tpc->SetBranchAddress("imposedTB",&bcTB);
    
    // scatter(tpc, "c1", "iniTrack", "iniTrackRef");


    // TCanvas c1 = difference1D(tpc, "c1","iniTrack","iniTrackRef"); 
    // sigma(tpc,"c2","iniTrack","iniTrackRef");

    TCanvas cim = iniMov(tpc,"cim","iniTrackRef","movTrackRef");



}