/** MLB team brand colors + names. Ported from the legacy `TEAM_BRAND` in src/ui_shared.py. */
export interface TeamBrand {
  abbr: string;
  name: string;
  primary: string;
  secondary: string;
}

export const TEAM_BRAND: Record<number, TeamBrand> = {
  108: { abbr: "LAA", name: "Los Angeles Angels", primary: "#BA0021", secondary: "#003263" },
  109: { abbr: "ARI", name: "Arizona Diamondbacks", primary: "#A71930", secondary: "#E3D4AD" },
  110: { abbr: "BAL", name: "Baltimore Orioles", primary: "#DF4601", secondary: "#000000" },
  111: { abbr: "BOS", name: "Boston Red Sox", primary: "#BD3039", secondary: "#0C2340" },
  112: { abbr: "CHC", name: "Chicago Cubs", primary: "#0E3386", secondary: "#CC3433" },
  113: { abbr: "CIN", name: "Cincinnati Reds", primary: "#C6011F", secondary: "#000000" },
  114: { abbr: "CLE", name: "Cleveland Guardians", primary: "#0C2340", secondary: "#E31937" },
  115: { abbr: "COL", name: "Colorado Rockies", primary: "#33006F", secondary: "#C4CED4" },
  116: { abbr: "DET", name: "Detroit Tigers", primary: "#0C2340", secondary: "#FA4616" },
  117: { abbr: "HOU", name: "Houston Astros", primary: "#002D62", secondary: "#EB6E1F" },
  118: { abbr: "KC", name: "Kansas City Royals", primary: "#004687", secondary: "#BD9B60" },
  119: { abbr: "LAD", name: "Los Angeles Dodgers", primary: "#005A9C", secondary: "#EF3E42" },
  120: { abbr: "WSH", name: "Washington Nationals", primary: "#AB0003", secondary: "#14225A" },
  121: { abbr: "NYM", name: "New York Mets", primary: "#002D72", secondary: "#FF5910" },
  133: { abbr: "ATH", name: "Athletics", primary: "#003831", secondary: "#EFB21E" },
  134: { abbr: "PIT", name: "Pittsburgh Pirates", primary: "#27251F", secondary: "#FDB827" },
  135: { abbr: "SD", name: "San Diego Padres", primary: "#2F241D", secondary: "#FFC425" },
  136: { abbr: "SEA", name: "Seattle Mariners", primary: "#0C2C56", secondary: "#005C5C" },
  137: { abbr: "SF", name: "San Francisco Giants", primary: "#FD5A1E", secondary: "#27251F" },
  138: { abbr: "STL", name: "St. Louis Cardinals", primary: "#C41E3A", secondary: "#0C2340" },
  139: { abbr: "TB", name: "Tampa Bay Rays", primary: "#092C5C", secondary: "#8FBCE6" },
  140: { abbr: "TEX", name: "Texas Rangers", primary: "#003278", secondary: "#C0111F" },
  141: { abbr: "TOR", name: "Toronto Blue Jays", primary: "#134A8E", secondary: "#1D2D5C" },
  142: { abbr: "MIN", name: "Minnesota Twins", primary: "#002B5C", secondary: "#D31145" },
  143: { abbr: "PHI", name: "Philadelphia Phillies", primary: "#E81828", secondary: "#002D72" },
  144: { abbr: "ATL", name: "Atlanta Braves", primary: "#CE1141", secondary: "#13274F" },
  145: { abbr: "CWS", name: "Chicago White Sox", primary: "#27251F", secondary: "#C4CED4" },
  146: { abbr: "MIA", name: "Miami Marlins", primary: "#00A3E0", secondary: "#EF3340" },
  147: { abbr: "NYY", name: "New York Yankees", primary: "#0C2340", secondary: "#C4CED4" },
  158: { abbr: "MIL", name: "Milwaukee Brewers", primary: "#12284B", secondary: "#FFC52F" },
};

export function teamBrand(teamId: number): TeamBrand {
  return (
    TEAM_BRAND[teamId] ?? {
      abbr: "MLB",
      name: "Free Agent",
      primary: "#0a1f3a",
      secondary: "#ff5c10",
    }
  );
}
