import argparse
import json
import os
import sys
import pathlib
import webbrowser
import pandas

folder = pathlib.Path(__file__).parent
while not folder.joinpath("chamois").exists():
    folder = folder.parent
sys.path.insert(0, str(folder))

import chamois.predictor

model = chamois.predictor.ChemicalOntologyPredictor.trained()

parser = argparse.ArgumentParser()
parser.add_argument("--report", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
parser.add_argument("--open", action="store_true")
args = parser.parse_args()

report = pandas.read_table(args.report)
schema = json.loads(
   """
	{
	  "$schema": "https://vega.github.io/schema/vega/v5.json",
	  "description": "An example of Cartesian layouts for a node-link diagram of hierarchical data.",
	  "width": 900,
	  "height": 1600,
	  "padding": 5,

	  "signals": [
		{
		  "name": "labels", "value": true,
		  "bind": {"input": "checkbox"}
		},
		{
		  "name": "layout", "value": "tidy",
		  "bind": {"input": "radio", "options": ["tidy", "cluster"]}
		},
		{
		  "name": "links", "value": "diagonal",
		  "bind": {
		    "input": "select",
		    "options": ["line", "curve", "diagonal", "orthogonal"]
		  }
		},
		{
		  "name": "separation", "value": false,
		  "bind": {"input": "checkbox"}
		}
	  ],

	  "data": [
		{
		  "name": "tree",
		  "values": [],
		  "transform": [
		    {
		      "type": "stratify",
		      "key": "id",
		      "parentKey": "parent"
		    },
		    {
		      "type": "tree",
		      "method": {"signal": "layout"},
		      "size": [{"signal": "height"}, {"signal": "width - 100"}],
		      "separation": {"signal": "separation"},
		      "as": ["y", "x", "depth", "children"]
		    }
		  ]
		},
		{
		  "name": "links",
		  "source": "tree",
		  "transform": [
		    { "type": "treelinks" },
		    {
		      "type": "linkpath",
		      "orient": "horizontal",
		      "shape": {"signal": "links"}
		    }
		  ]
		}
	  ],

	  "scales": [
		{
		  "name": "color",
		  "type": "linear",
                  "range": {"scheme": "turbo"},
		  "domain": {"data": "tree", "field": "adjusted_balanced_accuracy"},
		  "zero": true
		}
	  ],

	  "marks": [
		{
		  "type": "path",
		  "from": {"data": "links"},
		  "encode": {
		    "update": {
		      "path": {"field": "path"},
		      "stroke": {"value": "#c0c0c0"}
		    }
		  }
		},
		{
		  "type": "symbol",
		  "from": {"data": "tree"},
		  "encode": {
		    "enter": {
		      "size": {"value": 100},
		      "stroke": {"value": "#fff"}
		    },
		    "update": {
		      "x": {"field": "x"},
		      "y": {"field": "y"},
		      "fill": {"scale": "color", "field": "adjusted_balanced_accuracy"}
		    }
		  }
		},
		{
		  "type": "text",
		  "from": {"data": "tree"},
		  "encode": {
		    "enter": {
		      "text": {"field": "name"},
		      "fontSize": {"value": 9},
		      "baseline": {"value": "middle"}
		    },
		    "update": {
		      "x": {"field": "x"},
		      "y": {"field": "y"},
		      "dx": {"signal": "datum.children ? -7 : 7"},
		      "align": {"signal": "datum.children ? 'right' : 'left'"},
		      "opacity": {"signal": "labels ? 1 : 0"}
		    }
		  }
		}
	  ]
	}

   """
)
schema = json.loads(
    """
{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "description": "An example of a radial layout for a node-link diagram of hierarchical data.",
  "width": 1600,
  "height": 1600,
  "padding": 5,
  "autosize": "none",
  "usermeta": {
    "embedOptions": {
      "theme": "light",
      "loader": {"target": "_blank"}
    }
  },
  "legends": [
    {
      "fill": "color",
      "orient": "none",
      "legendX": 0,
      "legendY": 0,
      "padding": 20,
      "title": "AUPRC",
      "titleColor": "gray",
      "tickCount": 10,
      "tickMinStep": 0.2,
      "titleFontSize": 10
    }
  ],

  "signals": [
    {
      "name": "labels", "value": true,
      "bind": {"input": "checkbox"}
    },
    {
      "name": "radius", "value": 900,
      "bind": {"input": "range", "min": 20, "max": 1600}
    },
    {
      "name": "extent", "value": 360,
      "bind": {"input": "range", "min": 0, "max": 360, "step": 1}
    },
    {
      "name": "rotate", "value": 0,
      "bind": {"input": "range", "min": 0, "max": 360, "step": 1}
    },
    {
      "name": "layout", "value": "tidy",
      "bind": {"input": "radio", "options": ["tidy", "cluster"]}
    },
    {
      "name": "links", "value": "line",
      "bind": {
        "input": "select",
        "options": ["line", "curve", "diagonal", "orthogonal"]
      }
    },
    { "name": "originX", "update": "width / 2" },
    { "name": "originY", "update": "height / 2" }
  ],

  "data": [
    {
      "name": "tree",
      "values": [],
      "transform": [
        {
          "type": "stratify",
          "key": "id",
          "parentKey": "parent"
        },
        {
          "type": "tree",
          "method": {"signal": "layout"},
          "size": [1, {"signal": "radius"}],
          "as": ["alpha", "radius", "depth", "children"]
        },
        {
          "type": "formula",
          "expr": "(rotate + extent * datum.alpha + 270) % 360",
          "as":   "angle"
        },
        {
          "type": "formula",
          "expr": "PI * datum.angle / 180",
          "as":   "radians"
        },
        {
          "type": "formula",
          "expr": "inrange(datum.angle, [90, 270])",
          "as":   "leftside"
        },
        {
          "type": "formula",
          "expr": "originX + datum.radius * cos(datum.radians)",
          "as":   "x"
        },
        {
          "type": "formula",
          "expr": "originY + datum.radius * sin(datum.radians)",
          "as":   "y"
        }
      ]
    },
    {
      "name": "links",
      "source": "tree",
      "transform": [
        { "type": "treelinks" },
        {
          "type": "linkpath",
          "shape": {"signal": "links"}, "orient": "radial",
          "sourceX": "source.radians", "sourceY": "source.radius",
          "targetX": "target.radians", "targetY": "target.radius"
        }
      ]
    }
  ],

  "scales": [
    {
      "name": "color",
      "type": "linear",
      "range": {"scheme": "turbo"},
      "domain": {"data": "tree", "field": "auroc"},
      "zero": true
    }
  ],

  "marks": [
    {
      "type": "path",
      "from": {"data": "links"},
      "encode": {
        "update": {
          "x": {"signal": "originX"},
          "y": {"signal": "originY"},
          "path": {"field": "path"},
          "stroke": {"value": "#a0a0a0"}
        }
      }
    },
    {
      "type": "symbol",
      "from": {"data": "tree"},
      "encode": {
        "enter": {
          "size": {"value": 100},
          "stroke": {"value": "#fff"},
          "tooltip": {"signal": "{'class': datum.class, 'name': datum.name, 'ABA': datum.adjusted_balanced_accuracy, 'AUPRC': datum.auprc, 'AUPRC (Baseline)': datum.n_positives / 2046, 'AUROC': datum.auroc, 'Support': datum.n_positives}"},
          "href": {"signal": "\\"http://classyfire.wishartlab.com/tax_nodes/\\" + datum.class"}
        },
        "update": {
          "x": {"field": "x"},
          "y": {"field": "y"},
          "fill": {"scale": "color", "field": "auprc"}
        }
      }
    },
    {
      "type": "text",
      "from": {"data": "tree"},
      "encode": {
        "enter": {
          "text": {"field": "class"},
          "fontSize": {"value": 9},
          "baseline": {"value": "middle"}
        },
        "update": {
          "x": {"field": "x"},
          "y": {"field": "y"},
          "dx": {"signal": "(datum.leftside ? -1 : 1) * 6"},
          "angle": {"signal": "datum.leftside ? datum.angle - 180 : datum.angle"},
          "align": {"signal": "datum.leftside ? 'right' : 'left'"},
          "opacity": {"signal": "labels ? 1 : 0"}
        }
      }
    }
  ]
}
    """
)

root = {
    "id": len(model.classes_) + 1,
    "name": "Chemical Entitites",
    "class": "C9999999",
    "adusted_balanced_accuracy": 1.0,
}
schema["data"][0]["values"].append(root)

for row in report.itertuples():
    if row.n_positives < 10:
        continue
    # print(row)
    i = model.classes_.index.get_loc(row._4)
    datum = {
        "id": i,
        "name": row.name,
        "class": "C"+row._4.split(":")[1],
        "adjusted_balanced_accuracy": max(0.0, row.adjusted_balanced_accuracy),
        "auroc": row.auroc,
        "auprc": row.auprc,
        "n_positives": row.n_positives,
    }

    parents = model.ontology.adjacency_matrix.parents(i)
    if len(parents) > 0:
        datum["parent"] = int(parents[0])
    else:
        datum["parent"] = len(model.classes_) + 1
    schema["data"][0]["values"].append(datum)

    
    
with args.output.open("w") as dst:
    dst.write(
        """
        <head>
          <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
          <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
          <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        </head>
        <body>
          <div id="view"></div>
          <script>
          vegaEmbed( '#view', {} );
          </script>
        </body>
        """.format(json.dumps(schema, indent=4, sort_keys=True))
    )

if args.open:
    path = folder.joinpath("cvtree_auprc.html")
    webbrowser.open("file://{}".format(path))





