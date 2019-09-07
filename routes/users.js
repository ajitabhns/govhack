var express = require('express');
var router = express.Router();
var request = require('request');
var process = require('process');

process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = 0

/* GET users listing. */
// router.get('/', function(req, res, next) {
//   res.send('respond with a resource');
// });

/* GET users listing. */
router.get('/', function(req, res, next) {
  request({
    url: 'https://data.gov.au/data/api/3/action/datastore_search',
    qs: {
      resource_id:'23be1fc4-b6ef-4013-9102-c014c9d48711',
      q: '4 Alexander Avenue WENDOUREE VIC 3355'
    }
  }).on('error', error => {
res.status(502).send(error.message);
}).pipe(res);
});

module.exports = router;
