#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include "TMath.h"

using namespace std;

void scatter(TTree* tpc,const char* track1, const char* track2, Bool_t fSave = kFALSE, const char* cN = "c1", const char* file_ext = 0)
{
    TCanvas *c1 = new TCanvas(Form("%s",cN),Form("%s",cN),800,400);
    c1->Divide(3,2);
    gStyle->SetOptFit(1);
    gStyle->SetPalette(1);
    
    c1->cd(1);
    gPad->SetLogz();
    TF1 *pol1 = new TF1("pol1","pol1", 1, 0 ); 
    tpc->Draw(Form("%s_Z : %s_Z >> h1(100,-0.05,0.05,100,-0.05,0.05)",track1,track2),"","colz");
    auto h1 = (TH2F*)gDirectory->Get("h1");
    h1->Fit(pol1);


    c1->cd(2);
    gPad->SetLogz();
    tpc->Draw(Form("%s_Y : %s_Y >> h2(100,-0.03,0.03,100,-0.03,0.03)",track1,track2),"","colz");
    auto h2 = (TH2F*)gDirectory->Get("h2");
    h2->Fit(pol1);

    c1->cd(3);
    gPad->SetLogz();
    tpc->Draw(Form("%s_Snp : %s_Snp >> h3(100,-0.16,0.16,100,-0.16,0.16)",track1,track2),"","colz");
    auto h3 = (TH2F*)gDirectory->Get("h3");
    h3->Fit(pol1);

    c1->cd(4);
    gPad->SetLogz();
    tpc->Draw(Form("%s_Tgl : %s_Tgl >> h4(100,-0.036,0.036,100,-0.036,0.036)",track1,track2),"","colz");
    auto h4 = (TH2F*)gDirectory->Get("h4");
    h4->Fit(pol1);

    c1->cd(5);
    gPad->SetLogz();
    tpc->Draw(Form("%s_Q2Pt : %s_Q2Pt >> h5(100,-0.15,0.15,100,-0.15,0.15)",track1,track2),"","colz");
    auto h5 = (TH2F*)gDirectory->Get("h5");
    h5->Fit(pol1);

    if (fSave) c1->SaveAs(Form("plots_root/%s_vs_%s_%s.png",track1,track2,file_ext));
    
}

void correlation_of_ZDist_Tgl_Q2Pt(TTree* tpc, Bool_t fSave = kFALSE, const char* cN="c2", const char* file_ext = 0)
{
    TCanvas *c = new TCanvas(Form("%s",cN),Form("%s",cN));
    c->Divide(2,1);

    c->cd(1);
    tpc->Draw("movTrackRef_Z + dz - iniTrackRef_Z : iniTrackRef_Tgl : iniTrackRef_Q2Pt >> h2224510(500,0,50,500,0,3,500,-10,10)","","colz");

    c->cd(2);
    tpc->Draw("tNN_Z : iniTrackRef_Tgl : iniTrackRef_Q2Pt >> h222451(500,0,50,500,0,3,500,-10,10)","","colz");

    if (fSave) c->SaveAs(Form("plots_root/correlation_Z_Tgl_Q2Pt_%s.png",file_ext));

}

void sigmas(TTree* tpc, Bool_t fSave = kFALSE, TString dzCut = "", const char* cN="c3", const char* file_ext = 0)
{
    TCanvas *c = new TCanvas(Form("%s",cN),Form("%s",cN),800,400);
    c->Divide(3,2);
    gStyle->SetPalette(1);
    

    c->cd(1);
    tpc->Draw("TMath::Sqrt(TMath::Power(target_Z - NN_Z,2)) : target_Z : dz>> h45551(100,-100,100,100,-0.1,0.1,100,0,0.1)",dzCut,"colz");
    // gPad->SetLogy();
    // gPad->SetLogz();

    c->cd(2);
    tpc->Draw("TMath::Sqrt(TMath::Power(target_Y - NN_Y,2)) : target_Y : dz >> h45552(100,-100,100,100,0.2,0.2,100,0,0.2)",dzCut,"colz");
    // gPad->SetLogy();
    // gPad->SetLogz();

    c->cd(3);
    tpc->Draw("TMath::Sqrt(TMath::Power(target_Snp - NN_Snp,2)) : target_Snp : dz >> h45553(100,-100,100,100,-0.7,0.7,100,0.7,0.7)",dzCut,"colz");
    // gPad->SetLogy();
    // gPad->SetLogz();

    c->cd(4);
    tpc->Draw("TMath::Sqrt(TMath::Power(target_Tgl - NN_Tgl,2)) : target_Tgl : dz >> h45554(100,-100,100,100,-0.1,0.1,100,0,.1)",dzCut,"colz");
    // gPad->SetLogy();
    // gPad->SetLogz();

    c->cd(5);
    tpc->Draw("TMath::Sqrt(TMath::Power(target_Q2Pt - NN_Q2Pt,2)) : target_Q2Pt : dz >> h45555(100,-100,100,100,-0.6,0.6,100,0,0.6)",dzCut,"colz");
    // gPad->SetLogy();
    // gPad->SetLogz();

    if (fSave) c->SaveAs(Form("plots_root/sqrtdiff_%s.png",file_ext));

}

void dzplot(TTree *tpc,Bool_t fSave = kFALSE, const char* file_ext = 0)
{
    TCanvas *c123 = new TCanvas("c123","c123");
    c123->cd();
    gPad->SetLogy();
    tpc->Draw("dz");

    if (fSave) c123->SaveAs(Form("plots_root/dz_dist_%s.png",file_ext));
}


void tree_plots()//const char* inputFile)
{
    Bool_t fSave = kTRUE;

    TString plt_ext = "31082023_0007_positiveshift"; //
    
    auto *inputFile = TFile::Open("/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/eval/trees/TPC-SCD-NN-Prediction-FNet_0007_iniRef_Za0_Tgla0_dz_positiveshift_wClusters.root", "READ");
    auto *tpc = inputFile->Get<TTree>("tpc");

    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);

    scatter(tpc,"target","NN", fSave = fSave,"c1", plt_ext);
    correlation_of_ZDist_Tgl_Q2Pt(tpc, fSave = fSave,"c2", plt_ext);

    TString dzc = "";
    sigmas(tpc, fSave, dzc,"c3", plt_ext);
    dzplot(tpc, fSave, plt_ext);
    


}