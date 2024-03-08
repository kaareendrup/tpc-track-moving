#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include "TMath.h"

TCanvas *iniMov(TTree* tpc, const char* cc, const char* tr1, const char* tr2)
{
    TProfile *prof;

    gStyle->SetPalette(1);
    auto *c1 = new TCanvas(Form("%s",cc),Form("%s",cc),1400,600);

    c1->Divide(3,2);
    c1->cd(1);
    gPad->SetLogz();
    tpc->Draw(Form("%s->getY() - %s->getY() : %s->getY() >> h1(500,-120,120,500,-35,35)",tr1,tr2,tr1),"","colz");
    TH2 *h1 = (TH2*)gPad->GetPrimitive("h1");
    h1->GetYaxis()->SetTitle("Y_{ini} - Y_{ref}");
    h1->GetXaxis()->SetTitle("Y_{ini}");
    h1->SetTitle("Y");
    
    prof = h1->ProfileX("hprof1");
    prof->SetLineColor(kRed);
    prof->Draw("same");
    prof = nullptr;
    // 
    c1->cd(2);
    gPad->SetLogz();
    tpc->Draw(Form("%s->getZ() - %s->getZ() : %s->getZ() >> h2(500,-250,250,500,-25,25)",tr1,tr2,tr1),"","colz");
    TH2 *h2 = (TH2*)gPad->GetPrimitive("h2");
    h2->GetYaxis()->SetTitle("Z_{ini} - Z_{ref}");
    h2->GetXaxis()->SetTitle("Z_{ini}");
    h2->SetTitle("Z");

    prof = h2->ProfileX("hprof2");
    prof->SetLineColor(kRed);
    prof->Draw("same");
    prof = nullptr;
    // 
    c1->cd(3);
    gPad->SetLogz();
    tpc->Draw(Form("%s->getSnp() - %s->getSnp() : %s->getSnp() >> h3(500,-1,1,500,-1,1)",tr1,tr2,tr1),"","colz");
    TH2 *h3 = (TH2*)gPad->GetPrimitive("h3");
    h3->GetYaxis()->SetTitle("sin#phi_{ini} - sin#phi_{ref}");
    h3->GetXaxis()->SetTitle("sin#phi_{ini}");
    h3->SetTitle("sin#phi");

    prof = h3->ProfileX("hprof3");
    prof->SetLineColor(kRed);
    prof->Draw("same");
    prof = nullptr;
    // 
    c1->cd(4);
    gPad->SetLogz();
    tpc->Draw(Form("%s->getTgl() - %s->getTgl() : %s->getTgl() >> h4(500,-3,3,500,-0.4,0.4)",tr1,tr2,tr1),"","colz");
    TH2 *h4 = (TH2*)gPad->GetPrimitive("h4");
    h4->GetYaxis()->SetTitle("tan#lambda_{ini} - tan#lambda_{ref}");
    h4->GetXaxis()->SetTitle("tan#lambda_{ini}");
    h4->SetTitle("#lambda");

    prof = h4->ProfileX("hprof4");
    prof->SetLineColor(kRed);
    prof->Draw("same");
    prof = nullptr;
    // 
    c1->cd(5);
    gPad->SetLogz();
    tpc->Draw(Form("%s->getQ2Pt() - %s->getQ2Pt() : %s->getQ2Pt() >> h5(500,-40,40,500,-15,15)",tr1,tr2,tr1),"","colz");
    TH2 *h5 = (TH2*)gPad->GetPrimitive("h5");
    h5->GetYaxis()->SetTitle("(q/p_{T})_{ini} - (q/p_{T})_{ref}");
    h5->GetXaxis()->SetTitle("(q/p_{T})_{ini}");
    h5->SetTitle("q/p_{T}");

    prof = h5->ProfileX("hprof5");
    prof->SetLineColor(kRed);
    prof->Draw("same");
    prof = nullptr;

    return c1;
}

// void qa_plots(const char* inputfile)//, const char* savepath)
void qa_plots()
{
    
    // auto *inputFile = TFile::Open("/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/QA_tpctrack/tpc-trackStudy_newCM_6nm_n0.root", "READ");
    auto *inputFile = TFile::Open("/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/reconstructed_2024/tpc-trackStudy.root", "READ");
    
    auto *tpcIni = inputFile->Get<TTree>("tpcIni");

    // tpcIni->BuildIndex("counter","counter");
    // tpcMov->AddFriend(tpcIni);
    // auto *tpc = tpcMov;

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
    // TCanvas *csig = sigma(tpc,"c2","iniTrack","iniTrackRef");

    // TCanvas *cim = iniMov(tpc,"cim","iniTrackRef","movTrackRef");
    // cim->SaveAs("plots/iniMov_iniTrackRef_movTrackRef.png");

    auto c = iniMov(tpcIni,"c1","iniTrack","iniTrackRef");
    c->Draw();

    c->SaveAs("qa_plots_feb2024.pdf");

}