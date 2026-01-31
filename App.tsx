import { useState, useMemo } from "react";
import { DashboardHeader } from "./components/DashboardHeader";
import { ActionsList } from "./components/ActionsList";
import { MatchingPlatformMetrics } from "./components/MatchingPlatformMetrics";
import { SearchPage } from "./components/SearchPage";
import { MatchingPlatform } from "./components/MatchingPlatform";
import { VerificationPage } from "./components/VerificationPage";
import { BulkReclassificationModal } from "./components/BulkReclassificationModal";
import { MediaArticleDetailModal } from "./components/MediaArticleDetailModal";
import { CBSArticleDetailModal } from "./components/CBSArticleDetailModal";
import { MediaArticleMatch } from "./components/MatchingPlatform";
import { Toaster, toast } from "sonner@2.0.3";
import { initialMatchingArticles } from "./data/matchingArticlesData";

export default function App() {
  const [showSearchPage, setShowSearchPage] = useState(false);
  const [searchPageTab, setSearchPageTab] = useState<"cbs" | "media">("cbs");
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [selectedArticleForVerification, setSelectedArticleForVerification] = useState<MediaArticleMatch | null>(null);
  const [showBulkModal, setShowBulkModal] = useState(false);
  const [bulkModalData, setBulkModalData] = useState<{
    originalArticle: { title: string; oldCbsArticle: string; newCbsArticle: string };
    similarArticles: any[];
  } | null>(null);
  const [selectedMediaArticle, setSelectedMediaArticle] = useState<any>(null);
  const [selectedCbsArticle, setSelectedCbsArticle] = useState<any>(null);
  const [previousCbsArticle, setPreviousCbsArticle] = useState<any>(null);
  const [previousMediaArticle, setPreviousMediaArticle] = useState<any>(null);
  const [matchingArticles, setMatchingArticles] = useState<MediaArticleMatch[]>(initialMatchingArticles);
  const [activeFilter, setActiveFilter] = useState<"potentialMatches" | "orphans" | "deferred" | "errors" | "closed">("potentialMatches");
  const [showActionPage, setShowActionPage] = useState(false);

  // Mock data for recent articles
  const recentArticles = [
    {
      id: "1",
      title: "Werkloosheid blijft stabiel in oktober 2025",
      date: "22 nov 2025",
      category: "Arbeid en inkomen",
      summary: "De werkloosheid in Nederland blijft stabiel op 3.6% in oktober 2025. Dit blijkt uit de nieuwste cijfers van het CBS.",
    },
    {
      id: "2",
      title: "Inflatie daalt naar 3,2% in november",
      date: "21 nov 2025",
      category: "Economie",
      summary: "De inflatie in Nederland is in november gedaald naar 3,2%, een daling van 0,3 procentpunt ten opzichte van oktober.",
    },
    {
      id: "3",
      title: "Bevolkingsgroei zet door in stedelijke gebieden",
      date: "20 nov 2025",
      category: "Maatschappij",
      summary: "De Nederlandse bevolking blijft vooral groeien in stedelijke gebieden. Amsterdam, Rotterdam en Utrecht zien de grootste groei.",
    },
    {
      id: "4",
      title: "Economische groei trekt aan in derde kwartaal",
      date: "19 nov 2025",
      category: "Economie",
      summary: "Het bruto binnenlands product (bbp) groeide met 0,8% in het derde kwartaal van 2025.",
    },
    {
      id: "5",
      title: "Gezondheid Nederlanders verbetert licht",
      date: "18 nov 2025",
      category: "Maatschappij",
      summary: "De algemene gezondheid van Nederlanders toont een lichte verbetering volgens de laatste CBS gezondheidsenquête.",
    },
    {
      id: "6",
      title: "Woningprijzen stijgen met 4,5% in oktober",
      date: "17 nov 2025",
      category: "Economie",
      summary: "De gemiddelde verkoopprijs van woningen in Nederland is in oktober met 4,5% gestegen ten opzichte van een jaar eerder.",
    },
    {
      id: "7",
      title: "Aantal zzp'ers neemt verder toe",
      date: "16 nov 2025",
      category: "Arbeid en inkomen",
      summary: "Het aantal zelfstandigen zonder personeel (zzp'ers) is in het derde kwartaal van 2025 met 2,3% gegroeid.",
    },
    {
      id: "8",
      title: "CO2-uitstoot daalt met 6% in eerste drie kwartalen",
      date: "15 nov 2025",
      category: "Maatschappij",
      summary: "De CO2-uitstoot van Nederland is in de eerste drie kwartalen van 2025 met 6% gedaald ten opzichte van dezelfde periode in 2024.",
    },
    {
      id: "9",
      title: "Consumentenvertrouwen stijgt naar hoogste punt in twee jaar",
      date: "14 nov 2025",
      category: "Economie",
      summary: "Het vertrouwen van Nederlandse consumenten is in november gestegen naar het hoogste niveau sinds eind 2022.",
    },
    {
      id: "10",
      title: "Levensverwachting stijgt naar 82,3 jaar",
      date: "13 nov 2025",
      category: "Maatschappij",
      summary: "De gemiddelde levensverwachting in Nederland is gestegen naar 82,3 jaar, een stijging van 0,4 jaar ten opzichte van vorig jaar.",
    },
    {
      id: "11",
      title: "Detailhandelsomzet groeit met 3,7%",
      date: "12 nov 2025",
      category: "Economie",
      summary: "De omzet in de detailhandel is in oktober met 3,7% gestegen vergeleken met oktober 2024.",
    },
    {
      id: "12",
      title: "Aantal faillissementen neemt af",
      date: "11 nov 2025",
      category: "Economie",
      summary: "In oktober werden 315 bedrijven en instellingen failliet verklaard, een daling van 12% ten opzichte van september.",
    },
    {
      id: "13",
      title: "Gemiddeld inkomen huishoudens stijgt met 2,1%",
      date: "10 nov 2025",
      category: "Arbeid en inkomen",
      summary: "Het gemiddelde gestandaardiseerde inkomen van Nederlandse huishoudens is in 2024 met 2,1% gestegen naar 44.900 euro.",
    },
    {
      id: "14",
      title: "Aantal studenten in hoger onderwijs blijft groeien",
      date: "9 nov 2025",
      category: "Maatschappij",
      summary: "Het aantal studenten in het hoger onderwijs is dit studiejaar met 1,8% toegenomen tot ruim 920.000 studenten.",
    },
    {
      id: "15",
      title: "Energieverbruik huishoudens daalt met 8%",
      date: "8 nov 2025",
      category: "Maatschappij",
      summary: "Nederlandse huishoudens hebben in 2024 8% minder energie verbruikt dan in 2023, vooral door minder gasverbruik.",
    },
  ];

  // Mock data for recent media articles
  const recentMediaArticles = [
    {
      id: "MA-2025-001",
      title: "Werkloosheidscijfers blijven gunstig voor arbeidsmarkt",
      source: "NOS",
      sourceType: "online",
      date: "22 nov 2025",
      cbsArticle: "Werkloosheid blijft stabiel in oktober 2025",
      snippet: "Het CBS rapporteert dat de Nederlandse arbeidsmarkt stabiel blijft met een werkloosheidspercentage van 3.6%.",
      content: "AMSTERDAM - De werkloosheid in Nederland blijft op een historisch laag niveau staan. Volgens de nieuwste cijfers van het Centraal Bureau voor de Statistiek (CBS) bedroeg het werkloosheidspercentage in oktober 3.6%, wat nagenoeg gelijk is aan de voorgaande maanden.\\n\\nDe stabiele arbeidsmarkt komt mede door de aanhoudende vraag naar werknemers in verschillende sectoren. Met name de zorg, techniek en IT blijven kampen met personeelstekorten. Economen voorspellen dat de werkloosheid voorlopig op dit niveau zal blijven.\\n\\nDe cijfers tonen aan dat Nederland een van de laagste werkloosheidspercentages in Europa heeft. Dit heeft echter ook nadelen: veel bedrijven kunnen niet alle vacatures vervullen, wat groei soms belemmert.",
      link: "https://nos.nl/artikel/2025/werkloosheidscijfers-gunstig",
      page: "12",
      edition: "Ochtend"
    },
    {
      id: "MA-2025-002",
      title: "Prijsstijgingen nemen af volgens nieuwe cijfers",
      source: "De Telegraaf",
      sourceType: "print",
      date: "21 nov 2025",
      cbsArticle: "Inflatie daalt naar 3,2% in november",
      snippet: "De inflatie in Nederland daalt verder naar 3.2%, een welkome ontwikkeling voor consumenten.",
      content: "DEN HAAG - Voor het eerst in maanden is er goed nieuws voor de Nederlandse consument. De inflatie is in november gedaald naar 3.2%, een significante daling ten opzichte van de 3.5% van oktober. Dit blijkt uit cijfers van het CBS die gisteren zijn gepubliceerd.\\n\\nDe daling wordt voornamelijk veroorzaakt door lagere energieprijzen en een stabilisering van de voedselprijzen. Economen zien dit als een positief teken dat de prijsstijgingen eindelijk aan het afnemen zijn.\\n\\nToch waarschuwen experts dat de koopkracht van veel huishoudens nog altijd onder druk staat. De prijzen liggen nog steeds ver boven het niveau van twee jaar geleden.",
      link: "https://telegraaf.nl/artikel/prijsstijgingen-nemen-af",
      page: "1",
      edition: "Nationale editie"
    },
  ];

  const handleSearch = () => {
    setShowSearchPage(true);
  };

  const handleBackToDashboard = () => {
    setShowSearchPage(false);
    setSelectedArticleForVerification(null);
    setActiveFilter("potentialMatches");
    setShowActionPage(false);
  };

  // Filter articles based on active filter
  const filteredArticles = useMemo(() => {
    if (activeFilter === "potentialMatches") {
      // Potential matches: ≥85% confidence AND has a match AND not 100% confidence
      return matchingArticles.filter(a => a.matchedCbsArticle && a.confidenceScore >= 85 && a.confidenceScore < 100);
    }
    
    if (activeFilter === "orphans") {
      // Orphans: no match AND not 100% confidence
      return matchingArticles.filter(a => !a.matchedCbsArticle && a.confidenceScore < 100);
    }
    
    if (activeFilter === "deferred") {
      // Deferred: status = deferred AND not 100% confidence
      return matchingArticles.filter(a => a.verificationStatus === "deferred" && a.confidenceScore < 100);
    }
    
    if (activeFilter === "errors") {
      // Error matches: <85% confidence AND has a match AND not 100% confidence
      return matchingArticles.filter(a => a.matchedCbsArticle && a.confidenceScore < 85);
    }
    
    if (activeFilter === "closed") {
      // Closed: status = closed AND not 100% confidence
      return matchingArticles.filter(a => a.verificationStatus === "closed" && a.confidenceScore < 100);
    }
    
    return matchingArticles.filter(a => a.confidenceScore < 100);
  }, [matchingArticles, activeFilter]);

  // Calculate action counts (exclude 100% confidence)
  const actionCounts = useMemo(() => ({
    orphans: matchingArticles.filter(a => !a.matchedCbsArticle && a.confidenceScore < 100).length,
    deferred: matchingArticles.filter(a => a.verificationStatus === "deferred" && a.confidenceScore < 100).length,
    errors: matchingArticles.filter(a => a.matchedCbsArticle && a.confidenceScore < 85).length,
    closed: matchingArticles.filter(a => a.verificationStatus === "closed" && a.confidenceScore < 100).length,
  }), [matchingArticles]);

  // Calculate metrics from matchingArticles (exclude 100% confidence)
  const metrics = useMemo(() => {
    const allChildren = matchingArticles.filter(a => a.confidenceScore < 100);
    const potentialMatches = matchingArticles.filter(a => a.matchedCbsArticle && a.confidenceScore >= 85 && a.confidenceScore < 100);
    const orphans = matchingArticles.filter(a => !a.matchedCbsArticle && a.confidenceScore < 100);
    const deferred = matchingArticles.filter(a => a.verificationStatus === "deferred" && a.confidenceScore < 100);
    const errors = matchingArticles.filter(a => a.matchedCbsArticle && a.confidenceScore < 85);
    const closed = matchingArticles.filter(a => a.verificationStatus === "closed" && a.confidenceScore < 100);
    
    // Get unique CBS articles (parents)
    const uniqueParents = new Set(
      matchingArticles
        .filter(a => a.matchedCbsArticle && a.confidenceScore < 100)
        .map(a => a.matchedCbsArticle)
    );
    
    return {
      totalMatches: potentialMatches.length,
      verified: matchingArticles.filter(a => a.verificationStatus === "verified" && a.confidenceScore < 100).length,
      pending: matchingArticles.filter(a => a.verificationStatus === "pending" && a.confidenceScore < 100).length,
      automatic: matchingArticles.filter(a => a.confidenceScore >= 90 && a.confidenceScore < 100).length,
      manual: matchingArticles.filter(a => a.confidenceScore < 90).length,
      totalChildren: allChildren.length,
      totalParents: uniqueParents.size,
      inAfwachting: potentialMatches.length + orphans.length + deferred.length + errors.length
    };
  }, [matchingArticles]);

  const handleActionClick = (action: "orphans" | "deferred" | "errors" | "closed") => {
    setActiveFilter(action);
    setShowActionPage(true);
  };

  const handleSelectArticleForVerification = (article: MediaArticleMatch) => {
    setSelectedArticleForVerification(article);
  };

  const handleBackToMatchingPlatform = () => {
    setSelectedArticleForVerification(null);
  };

  const handleReclassify = (articleId: string, newCategory: string, newCbsArticle: string) => {
    // Find similar articles based on the article being reclassified
    let similarArticles = [];
    
    if (selectedArticleForVerification?.title === "Scholieren presteren beter op wiskundetoets") {
      // Articles related to education/Onderwijsmonitor 2025
      const educationArticles = matchingArticles.filter(a => 
        ["sim1", "sim2", "sim3", "sim4"].includes(a.id)
      );
      
      similarArticles = educationArticles.map(article => ({
        id: article.id,
        title: article.title,
        source: article.source,
        confidenceScore: article.confidenceScore,
        currentCbsArticle: article.matchedCbsArticle
      }));
    } else {
      // Default energy-related articles for other cases
      similarArticles = [
        {
          id: "sim1",
          title: "Energiekosten blijven hoog voor bedrijven",
          source: "NRC",
          confidenceScore: 62,
          currentCbsArticle: "Energiemarkt Q4 2025"
        },
        {
          id: "sim2",
          title: "Duurzame energie groeit met 10%",
          source: "Trouw",
          confidenceScore: 71,
          currentCbsArticle: "Energiemarkt Q4 2025"
        },
        {
          id: "sim3",
          title: "Windmolens produceren recordhoeveelheid stroom",
          source: "De Volkskrant",
          confidenceScore: 68,
          currentCbsArticle: "Energiemarkt Q4 2025"
        }
      ];
    }

    if (selectedArticleForVerification) {
      setBulkModalData({
        originalArticle: {
          title: selectedArticleForVerification.title,
          oldCbsArticle: selectedArticleForVerification.matchedCbsArticle,
          newCbsArticle: newCbsArticle
        },
        similarArticles
      });
      setShowBulkModal(true);
    }
  };

  const handleBulkReclassify = (selectedArticleIds: string[]) => {
    if (!selectedArticleForVerification || !bulkModalData) return;

    // Update the original article and selected similar articles
    const articleIdsToUpdate = [selectedArticleForVerification.id, ...selectedArticleIds];
    const newCbsArticle = bulkModalData.originalArticle.newCbsArticle;

    setMatchingArticles(prevArticles => 
      prevArticles.map(article => {
        if (articleIdsToUpdate.includes(article.id)) {
          return {
            ...article,
            matchedCbsArticle: newCbsArticle,
            confidenceScore: 100,
            verificationStatus: "verified" as const
          };
        }
        return article;
      })
    );

    // Show success toast
    const count = selectedArticleIds.length + 1; // +1 for original article
    toast.success(`${count} artikel${count > 1 ? 'en' : ''} succesvol gematcht`, {
      description: `Gekoppeld aan "${newCbsArticle}". Metrics zijn bijgewerkt.`,
      duration: 4000,
    });
    
    // Close modal and go back to matching platform
    setShowBulkModal(false);
    setSelectedArticleForVerification(null);
  };

  const handleBulkVerify = (cbsArticle: string, articleIds: string[]) => {
    setMatchingArticles(prevArticles => 
      prevArticles.map(article => {
        if (articleIds.includes(article.id)) {
          return {
            ...article,
            verificationStatus: "verified" as const,
            confidenceScore: 100 // Set to 100% because match is confirmed
          };
        }
        return article;
      })
    );

    // Show success toast
    toast.success(`${articleIds.length} artikel${articleIds.length > 1 ? 'en' : ''} succesvol geverifieerd`, {
      description: `Gekoppeld aan "${cbsArticle}". Betrouwbaarheid ingesteld op 100%.`,
      duration: 4000,
    });
  };

  const handleCbsArticleClick = (cbsArticleTitle: string) => {
    // Find the CBS article by title
    const article = recentArticles.find(a => a.title === cbsArticleTitle);
    if (article) {
      // Save the current media article as previous before opening CBS article
      if (selectedMediaArticle) {
        setPreviousMediaArticle(selectedMediaArticle);
      }
      setSelectedCbsArticle(article);
    }
  };

  const handleMediaArticleClick = (article: any) => {
    // Save the current CBS article as previous before opening media article
    if (selectedCbsArticle) {
      setPreviousCbsArticle(selectedCbsArticle);
    }
    setSelectedMediaArticle(article);
  };

  const handleBackToPreviousCbsArticle = () => {
    setSelectedMediaArticle(null);
    if (previousCbsArticle) {
      setSelectedCbsArticle(previousCbsArticle);
      setPreviousCbsArticle(null);
    }
  };

  const handleBackToPreviousMediaArticle = () => {
    setSelectedCbsArticle(null);
    if (previousMediaArticle) {
      setSelectedMediaArticle(previousMediaArticle);
      setPreviousMediaArticle(null);
    }
  };

  // Show verification page
  if (selectedArticleForVerification) {
    return (
      <>
        <Toaster position="bottom-right" theme={isDarkMode ? "dark" : "light"} />
        <VerificationPage
          article={selectedArticleForVerification}
          onBack={handleBackToMatchingPlatform}
          onBackToDashboard={handleBackToDashboard}
          isDarkMode={isDarkMode}
          onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
          onReclassify={handleReclassify}
        />
        {bulkModalData && (
          <BulkReclassificationModal
            open={showBulkModal}
            onClose={() => setShowBulkModal(false)}
            onConfirm={handleBulkReclassify}
            originalArticle={bulkModalData.originalArticle}
            similarArticles={bulkModalData.similarArticles}
            isDarkMode={isDarkMode}
          />
        )}
      </>
    );
  }

  // Show action page (full-screen matching platform for specific action)
  if (showActionPage && activeFilter !== "potentialMatches") {
    return (
      <>
        <Toaster position="bottom-right" theme={isDarkMode ? "dark" : "light"} />
        <MatchingPlatform
          onBack={() => {
            setShowActionPage(false);
            setActiveFilter("potentialMatches");
          }}
          isDarkMode={isDarkMode}
          onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
          onSelectArticle={handleSelectArticleForVerification}
          onMediaArticleClick={setSelectedMediaArticle}
          onCbsArticleClick={handleCbsArticleClick}
          articles={filteredArticles}
          hideHeader={false}
          activeFilter={activeFilter}
        />
        <MediaArticleDetailModal
          article={selectedMediaArticle}
          onClose={() => setSelectedMediaArticle(null)}
          isDarkMode={isDarkMode}
          onCbsArticleClick={handleCbsArticleClick}
          onBack={previousCbsArticle ? handleBackToPreviousCbsArticle : undefined}
        />
        <CBSArticleDetailModal
          article={selectedCbsArticle}
          onClose={() => setSelectedCbsArticle(null)}
          isDarkMode={isDarkMode}
          onMediaArticleClick={handleMediaArticleClick}
          onBack={previousMediaArticle ? handleBackToPreviousMediaArticle : undefined}
        />
      </>
    );
  }

  // Show search page
  if (showSearchPage) {
    return (
      <>
        <Toaster position="bottom-right" theme={isDarkMode ? "dark" : "light"} />
        <SearchPage 
          onBack={handleBackToDashboard}
          cbsArticles={recentArticles}
          mediaArticles={recentMediaArticles}
          tab={searchPageTab}
          onTabChange={setSearchPageTab}
          isDarkMode={isDarkMode}
          onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
          onCbsArticleClick={(article) => setSelectedCbsArticle(article)}
          onMediaArticleClick={(article) => setSelectedMediaArticle(article)}
        />
        <MediaArticleDetailModal
          article={selectedMediaArticle}
          onClose={() => setSelectedMediaArticle(null)}
          isDarkMode={isDarkMode}
          onCbsArticleClick={handleCbsArticleClick}
          onBack={previousCbsArticle ? handleBackToPreviousCbsArticle : undefined}
        />
        <CBSArticleDetailModal
          article={selectedCbsArticle}
          onClose={() => setSelectedCbsArticle(null)}
          isDarkMode={isDarkMode}
          onMediaArticleClick={handleMediaArticleClick}
          onBack={previousMediaArticle ? handleBackToPreviousMediaArticle : undefined}
        />
      </>
    );
  }

  // Show dashboard with Matching Platform on the foreground
  return (
    <div className={`h-screen flex flex-col overflow-hidden ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <Toaster position="bottom-right" theme={isDarkMode ? "dark" : "light"} />
      <DashboardHeader 
        onSearchClick={handleSearch}
        isDarkMode={isDarkMode}
        onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
      />
      
      <main className="max-w-[1880px] mx-auto px-8 py-4 flex-1 w-full overflow-hidden">
        <div className="grid gap-4 h-full" style={{ gridTemplateColumns: '1fr 320px' }}>
          {/* Matching Platform - Main Content */}
          <div className="h-full overflow-hidden">
            <MatchingPlatform
              onBack={handleBackToDashboard}
              isDarkMode={isDarkMode}
              onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
              onSelectArticle={handleSelectArticleForVerification}
              onMediaArticleClick={setSelectedMediaArticle}
              onCbsArticleClick={handleCbsArticleClick}
              articles={filteredArticles}
              hideHeader={true}
              activeFilter={activeFilter}
              onClearFilter={() => setActiveFilter("potentialMatches")}
              onBulkVerify={handleBulkVerify}
            />
          </div>
          
          {/* Actions List - Right Sidebar */}
          <div className="h-full overflow-hidden flex flex-col gap-4">
            <div className="flex-none">
              <MatchingPlatformMetrics 
                isDarkMode={isDarkMode}
                metrics={metrics}
              />
            </div>
            <div className="flex-1 overflow-hidden">
              <ActionsList 
                isDarkMode={isDarkMode}
                onActionClick={handleActionClick}
                counts={actionCounts}
              />
            </div>
          </div>
        </div>
      </main>

      <MediaArticleDetailModal
        article={selectedMediaArticle}
        onClose={() => setSelectedMediaArticle(null)}
        isDarkMode={isDarkMode}
        onCbsArticleClick={handleCbsArticleClick}
        onBack={previousCbsArticle ? handleBackToPreviousCbsArticle : undefined}
      />
      <CBSArticleDetailModal
        article={selectedCbsArticle}
        onClose={() => setSelectedCbsArticle(null)}
        isDarkMode={isDarkMode}
        onMediaArticleClick={handleMediaArticleClick}
        onBack={previousMediaArticle ? handleBackToPreviousMediaArticle : undefined}
      />
    </div>
  );
}